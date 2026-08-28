# -*- coding: utf-8 -*-
"""
evaluate_refcoco_checkpoint.py
-------------------------------
Standalone RefCOCO+ evaluation — the ONLY place real generation-based IoU@0.5 is computed.
Run after training, before destroying the instance.

Built the way evaluate_checkpoint.py (the VQA version) already does it correctly: rebuild a
fresh prompt-only input via apply_chat_template(add_generation_prompt=True) for every sample,
set padding_side="left" for generation, and never feed the model its own ground-truth answer.
This is deliberately independent of the training-time collator/prediction_step path.

Reports accuracy@0.5, mean IoU, and parse-failure rate separately for testA and testB — the
standard RefCOCO+ benchmark convention (testA = multi-person images, testB = other objects).

Usage:
    export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
    python evaluate_refcoco_checkpoint.py
"""

import os
import sys
import json
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from transformers import MllamaForConditionalGeneration, AutoProcessor
from peft import PeftModel

from data_utils_RefCOCOplus_Llama import (
    load_refcocoplus_benchmark_splits, box_xywh_to_norm, SYSTEM_MESSAGE, _pick_expression,
)
from metrics_refcoco import iou_accuracy, compute_iou
from data_utils_RefCOCOplus_Llama import parse_box

RUN_DIR        = os.environ.get(
    "REFCOCO_OUTPUT_DIR", "./results_refcocoplus_grit_llama3_2_vision")
FINAL_MODEL_DIR = f"{RUN_DIR}_final_model"
MODEL_ID       = "meta-llama/Llama-3.2-11B-Vision-Instruct"
HF_TOKEN       = os.environ.get("HF_TOKEN")
MAX_NEW_TOKENS = 32          # a box is short; no need for VQA's 128
NUM_EVAL_SAMPLES = int(os.environ.get("NUM_EVAL_SAMPLES", 250))  # per split (testA, testB)
OUTPUT_JSON    = os.environ.get(
    "REFCOCO_EVAL_OUTPUT_JSON", os.path.expanduser("~/eval_refcoco_results.json"))


def resolve_checkpoint_dir() -> str:
    """
    train_refcoco_grit.py's Trainer checkpoints are named checkpoint-<step> (never
    "checkpoint-best" — that was a placeholder bug in an earlier draft of this file).
    Prefer FINAL_MODEL_DIR (written after a completed run, and — because
    load_best_model_at_end=True — already IS the best checkpoint by eval_loss).
    Fall back to the highest-numbered checkpoint under RUN_DIR for an interrupted run.
    """
    if os.path.isdir(FINAL_MODEL_DIR):
        return FINAL_MODEL_DIR
    if os.path.isdir(RUN_DIR):
        candidates = [os.path.join(RUN_DIR, d) for d in os.listdir(RUN_DIR)
                      if d.startswith("checkpoint-") and d.rsplit("-", 1)[-1].isdigit()]
        if candidates:
            return max(candidates, key=lambda p: int(p.rsplit("-", 1)[-1]))
    raise FileNotFoundError(
        f"No checkpoint found. Looked for {FINAL_MODEL_DIR} and checkpoint-* under {RUN_DIR}. "
        "Set CHECKPOINT_DIR manually if your run used a different output_dir.")


CHECKPOINT_DIR = resolve_checkpoint_dir()


def load_model_and_processor():
    if not HF_TOKEN:
        raise EnvironmentError("HF_TOKEN not set. Run: export HF_TOKEN=hf_xxxxxxxxxxxxxxxx")

    processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    # Left padding for generation — right-padding here would misalign batched generate()
    # exactly as described under finding C3 in the code review.
    processor.tokenizer.padding_side = "left"

    # bf16, no quantization — must match how the base model was loaded during training,
    # otherwise the LoRA adapter (trained against bf16 activations) sees a distribution shift.
    base_model = MllamaForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16,
        device_map="auto", token=HF_TOKEN,
    )
    model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)
    model.eval()
    model.config.use_cache = True
    return model, processor


def generate_for_split(model, processor, dataset, split_name: str, n: int):
    predictions, gt_boxes, expressions = [], [], []
    n = min(n, len(dataset))

    for i in range(n):
        ex = dataset[i]
        try:
            image = ex["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")
            expression = _pick_expression(ex)
            bbox = ex.get("bbox")
            if not expression or bbox is None:
                continue

            img_w, img_h = image.size
            gt_box = box_xywh_to_norm(bbox, img_w, img_h)

            user_text = f'Referring expression: "{expression}"\nOutput the bounding box.'
            messages = [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
            ]
            prompt_text = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False)

            enc = processor(images=[image], text=[prompt_text], return_tensors="pt", padding=True)
            enc = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in enc.items()}

            input_len = enc["input_ids"].shape[1]  # exact with left-padding: real content ends here
            gen_kwargs = {k: v for k, v in enc.items()
                          if k in ["pixel_values", "image_sizes", "aspect_ratio_ids",
                                   "aspect_ratio_mask", "cross_attention_mask"]}

            with torch.no_grad():
                output_ids = model.generate(
                    enc["input_ids"], attention_mask=enc["attention_mask"],
                    max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    use_cache=True, **gen_kwargs,
                )
            new_tokens = output_ids[:, input_len:]
            pred_text = processor.tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()

            predictions.append(pred_text)
            gt_boxes.append(gt_box)
            expressions.append(expression)

            if (i + 1) % 25 == 0:
                print(f"  [{split_name}] {i+1}/{n}  pred={pred_text[:40]!r}  gt={gt_box}")

        except Exception as e:
            print(f"  [WARN] {split_name} sample {i} failed: {e}")

    return predictions, gt_boxes, expressions


def main():
    print("="*60)
    print("REFCOCO+ CHECKPOINT EVALUATION")
    print(f"Checkpoint: {CHECKPOINT_DIR}")
    print("="*60)

    model, processor = load_model_and_processor()
    testA, testB = load_refcocoplus_benchmark_splits()

    results = {}
    all_json = {}
    for split_name, ds in (("testA", testA), ("testB", testB)):
        print(f"\nGenerating on {split_name} ({min(NUM_EVAL_SAMPLES, len(ds))} samples)...")
        preds, gts, exprs = generate_for_split(model, processor, ds, split_name, NUM_EVAL_SAMPLES)
        stats = iou_accuracy(preds, gts, threshold=0.5)
        results[split_name] = stats
        all_json[split_name] = {
            "stats": stats,
            "samples": [{"expression": e, "prediction": p, "gt_box_norm": list(g)}
                        for e, p, g in zip(exprs, preds, gts)],
        }
        print(f"[{split_name}] accuracy@0.5={stats['accuracy_at_0.5']:.4f}  "
              f"mean_iou={stats['mean_iou']:.4f}  "
              f"parse_failures={stats['parse_failure_rate']:.4f}")

        # Text examples: a few in generation order, plus the worst-scoring ones (excluding
        # outright parse failures, which are a separate, already-reported failure mode).
        scored = []
        for e, p, g in zip(exprs, preds, gts):
            pb = parse_box(p)
            iou = compute_iou(pb, g) if pb is not None else None
            scored.append((e, p, pb, g, iou))

        print(f"\n  --- {split_name}: first {min(5, len(scored))} examples ---")
        for e, p, pb, g, iou in scored[:5]:
            iou_str = f"{iou:.3f}" if iou is not None else "PARSE FAILURE"
            print(f"  \"{e}\"\n    pred={p!r} -> box={pb}  gt={g}  IoU={iou_str}")

        worst = sorted([s for s in scored if s[4] is not None], key=lambda s: s[4])[:3]
        if worst:
            print(f"\n  --- {split_name}: 3 worst-scoring (excluding parse failures) ---")
            for e, p, pb, g, iou in worst:
                print(f"  \"{e}\"\n    pred={p!r} -> box={pb}  gt={g}  IoU={iou:.3f}")

    print("\n" + "="*60)
    print("FINAL RESULTS (real generation, not teacher-forced)")
    for split_name, stats in results.items():
        print(f"  {split_name}: acc@0.5={stats['accuracy_at_0.5']:.4f} "
              f"mean_iou={stats['mean_iou']:.4f} "
              f"parse_fail={stats['parse_failure_rate']:.4f} "
              f"(n={stats['num_samples']})")
    print("="*60)

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_json, f, indent=2)
    print(f"\nFull results saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
