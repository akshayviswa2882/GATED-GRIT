# -*- coding: utf-8 -*-
"""
train_refcoco_grit.py — GRIT fine-tuning for Llama-3.2-11B-Vision-Instruct on RefCOCO+ (REC)

This reuses grit_llama.py — GRITManager/GritTrainer/GritOptimizer only ever touch LoRA A/B
tensors and their K-FAC statistics, so nothing about them is VQA- or RefCOCO+-specific. Only
the data layer (data_utils_RefCOCOplus_Llama.py) and metrics (metrics_refcoco.py) are new.

Fixes applied relative to grit_llama_v2.py (see GRIT_code_review.md):
  C1 — tokenize_refcoco() never returns None; it returns {"_skip": True} and we .filter() on
       that flag, which IS supported by Dataset.map(), instead of crashing on the first bad row.
  C2 — resume_from_checkpoint is only passed if the path actually exists.
  C3 — predict_with_generate is never turned on during training. Only a safe teacher-forced
       proxy metric runs in-loop (metrics_refcoco.make_compute_metrics); real IoU@0.5 generation
       lives in evaluate_refcoco_checkpoint.py, built correctly from a fresh prompt-only input.
  H2 — LoRA target_modules are resolved by walking named_modules() and keeping only paths under
       "language_model." by default, instead of relying on name collisions with the vision tower.

Additional fixes to grit_llama.py itself, confirmed via smoke testing (see
smoke_test/README_SMOKE_TEST.md for how these were isolated):
  - GritTrainer.compute_loss(): nn.ModuleDict has no .get() — was crashing on step 1.
  - GritTrainer.prediction_step(): no longer calls model.generate() during in-training eval
    (metrics_refcoco.py's teacher-forced proxy expects same-length logits, not generated
    tokens — this was crashing the first time evaluation ran).
  - KFACAutogradFunction.backward(): a_cov_sample/g_cov_sample are now normalized by row
    count instead of being a raw unnormalized sum — this was inflating the curvature
    regularizer in compute_loss() to dominate the real CE loss by up to ~30x.
  - GritTrainer.training_step(): precondition_gradients() is now called directly here
    (gated on accelerator.sync_gradients) instead of from optimizer_step()/GritOptimizer,
    which turned out to be dead code never invoked by this transformers version — natural-
    gradient preconditioning was silently never running in any prior training attempt.

DATASET SIZE
------------
By default this trains on the full ~70% pool split (see load_refcocoplus_pool). To run on a
smaller/larger slice for faster iteration, set TRAIN_N / VAL_N:
    TRAIN_N=500 VAL_N=100 python train_refcoco_grit.py
Leave unset to use the full pool.

QUALITATIVE PREVIEW
--------------------
After training, this script runs a small number of real generations (default 8, see
NUM_QUALITATIVE_EXAMPLES) on held-out dev-test examples and prints/saves them as text
examples (expression, predicted box, ground-truth box, IoU) — a quick sanity check before
running the full evaluate_refcoco_checkpoint.py benchmark on testA/testB.
"""

import os
import sys
import gc
import json
import math
import time
import torch

# Make sure this script's own directory is importable regardless of cwd — needed for
# `from grit_llama import ...` / `from data_utils_RefCOCOplus_Llama import ...` below.
# All five files (this one, grit_llama.py, data_utils_RefCOCOplus_Llama.py, metrics_refcoco.py,
# evaluate_refcoco_checkpoint.py) must live in the same directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from transformers import (
    MllamaForConditionalGeneration,
    AutoProcessor,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model

from grit_llama import GRITConfig, GRITManager, GritTrainer, GritCallback
from data_utils_RefCOCOplus_Llama import (
    load_refcocoplus_pool, tokenize_refcoco, collate_fn,
    box_xywh_to_norm, SYSTEM_MESSAGE, _pick_expression,
)
from metrics_refcoco import make_compute_metrics, compute_iou
from data_utils_RefCOCOplus_Llama import parse_box

NUM_QUALITATIVE_EXAMPLES = int(os.environ.get("NUM_QUALITATIVE_EXAMPLES", 8))


def print_qualitative_examples(model, processor, dataset, n, output_dir):
    """
    Run a handful of REAL generations (never teacher-forced) on held-out examples and print
    them as readable text: expression -> predicted box vs. ground-truth box -> IoU. Saved
    alongside the checkpoint as qualitative_examples.json. This is a quick sanity check, not
    a benchmark — for real testA/testB numbers, run evaluate_refcoco_checkpoint.py.
    """
    print("\n" + "=" * 70)
    print(f"QUALITATIVE PREVIEW — {min(n, len(dataset))} real generations "
          f"(not teacher-forced)")
    print("=" * 70)

    model.eval()
    was_use_cache = model.config.use_cache
    model.config.use_cache = True

    examples = []
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
            input_len = enc["input_ids"].shape[1]
            gen_kwargs = {k: v for k, v in enc.items()
                          if k in ["pixel_values", "image_sizes", "aspect_ratio_ids",
                                   "aspect_ratio_mask", "cross_attention_mask"]}
            with torch.no_grad():
                output_ids = model.generate(
                    enc["input_ids"], attention_mask=enc["attention_mask"],
                    max_new_tokens=32, do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    use_cache=True, **gen_kwargs,
                )
            pred_text = processor.tokenizer.decode(
                output_ids[0, input_len:], skip_special_tokens=True).strip()
            pred_box = parse_box(pred_text)
            iou = compute_iou(pred_box, gt_box) if pred_box is not None else 0.0

            print(f"\n[{i}] \"{expression}\"")
            print(f"    predicted: {pred_text!r}  -> box={pred_box}")
            print(f"    ground truth box: {gt_box}")
            print(f"    IoU: {iou:.3f}  {'PASS (>=0.5)' if iou >= 0.5 else 'below threshold'}")

            examples.append({
                "expression": expression, "prediction_text": pred_text,
                "predicted_box": list(pred_box) if pred_box else None,
                "gt_box": list(gt_box), "iou": iou,
            })
        except Exception as e:
            print(f"  [WARN] qualitative example {i} failed: {e}")

    if examples:
        ious = [e["iou"] for e in examples]
        print(f"\n  Mean IoU over these {len(examples)} examples: {sum(ious)/len(ious):.3f}")
        print(f"  {sum(1 for x in ious if x >= 0.5)}/{len(examples)} at/above IoU 0.5")

    out_path = os.path.join(output_dir, "qualitative_examples.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(examples, f, indent=2)
    print(f"\n  Saved to {out_path}")
    print("=" * 70)

    model.config.use_cache = was_use_cache
    model.train()


def main():
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        raise EnvironmentError("HF_TOKEN not set. Run: export HF_TOKEN=hf_xxxxxxxxxxxxxxxx")

    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    config = GRITConfig()
    config.model_id   = model_id
    config.max_length = 512
    config.num_epochs = 3
    config.batch_size = 1
    config.gradient_accumulation_steps = 8
    config.learning_rate = 1e-4
    config.kfac_damping_ceiling = float(os.environ.get("KFAC_DAMPING_CEILING", 1.0))

    # GRIT activation thresholds — defaults match the original design (tuned for a full-
    # size dataset, where these are reached naturally within the first several dozen
    # steps). Override for a fast sanity check that forces preconditioning/reprojection to
    # actually fire within a handful of steps, e.g.:
    #   KFAC_UPDATE_FREQ=5 KFAC_MIN_SAMPLES=16 REPROJECTION_WARMUP=8 REPROJECTION_FREQ=8 \
    #   NG_WARMUP_STEPS=5 COV_UPDATE_FREQ=1 TRAIN_N=64 VAL_N=16 python train_refcoco_grit.py
    if "KFAC_UPDATE_FREQ" in os.environ:
        config.kfac_update_freq = int(os.environ["KFAC_UPDATE_FREQ"])
    if "KFAC_MIN_SAMPLES" in os.environ:
        config.kfac_min_samples = int(os.environ["KFAC_MIN_SAMPLES"])
    if "COV_UPDATE_FREQ" in os.environ:
        config.grit_cov_update_freq = int(os.environ["COV_UPDATE_FREQ"])
    if "REPROJECTION_WARMUP" in os.environ:
        config.reprojection_warmup_steps = int(os.environ["REPROJECTION_WARMUP"])
    if "REPROJECTION_FREQ" in os.environ:
        config.reprojection_freq = int(os.environ["REPROJECTION_FREQ"])
    if "NG_WARMUP_STEPS" in os.environ:
        config.ng_warmup_steps = int(os.environ["NG_WARMUP_STEPS"])
    print(f"GRIT thresholds: kfac_update_freq={config.kfac_update_freq} "
          f"kfac_min_samples={config.kfac_min_samples} "
          f"grit_cov_update_freq={config.grit_cov_update_freq} "
          f"reprojection_warmup_steps={config.reprojection_warmup_steps} "
          f"reprojection_freq={config.reprojection_freq} "
          f"ng_warmup_steps={config.ng_warmup_steps}")
    # FIX: GRITConfig.telemetry_dir defaults to a leftover VQA-project path name never
    # overridden here — telemetry.json (consumed by plot_heatmaps.py/plot_spectra.py/
    # plot_spectra_compare.py) was landing somewhere unrelated to this actual run.
    RUN_OUTPUT_DIR = os.environ.get(
        "REFCOCO_OUTPUT_DIR", "./results_refcocoplus_grit_llama3_2_vision")
    config.telemetry_dir = RUN_OUTPUT_DIR

    # Ablation switches for isolating which GRIT mechanism drives any observed effect.
    # Examples:
    #   Pure LoRA baseline, fastest:
    #     DISABLE_PRECONDITION=1 DISABLE_REPROJECTION=1 python train_refcoco_grit.py
    #   Pure LoRA baseline, but still measure curvature for spectral comparison:
    #     DISABLE_PRECONDITION=1 DISABLE_REPROJECTION=1 TRACK_COVARIANCE=1 python ...
    #   Preconditioning only, no reprojection/pruning:
    #     DISABLE_REPROJECTION=1 python train_refcoco_grit.py
    #   Reprojection only, no natural-gradient step:
    #     DISABLE_PRECONDITION=1 python train_refcoco_grit.py
    _flag = lambda name: os.environ.get(name, "0") == "1"
    config.disable_precondition = _flag("DISABLE_PRECONDITION")
    config.disable_reprojection = _flag("DISABLE_REPROJECTION")
    config.disable_curvature_regularizer = _flag("DISABLE_CURVATURE_REG")
    config.disable_reprojection_regularizer = _flag("DISABLE_REPROJECTION_REG")
    config.track_covariance_for_analysis = _flag("TRACK_COVARIANCE")
    print(f"GRIT ablation config: disable_precondition={config.disable_precondition} "
          f"disable_reprojection={config.disable_reprojection} "
          f"disable_curvature_regularizer={config.disable_curvature_regularizer} "
          f"disable_reprojection_regularizer={config.disable_reprojection_regularizer} "
          f"track_covariance_for_analysis={config.track_covariance_for_analysis}")
    # K-FAC / reprojection schedule left at GRITConfig defaults — unchanged from the VQA project.

    # -------------------------------------------------------------------
    # 1. Processor
    # -------------------------------------------------------------------
    processor = AutoProcessor.from_pretrained(model_id, token=HF_TOKEN, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    # Required for correct label masking — see tokenize_refcoco / the C3 discussion in the review.
    processor.tokenizer.padding_side = "right"

    # -------------------------------------------------------------------
    # 2. Dataset — real embedded images, no COCO-lookup/placeholder problem (fixes H3).
    # -------------------------------------------------------------------
    print("Loading RefCOCO+ pool (val split, carved into train/val/dev-test)...")
    train_raw, val_raw, devtest_raw = load_refcocoplus_pool(seed=42)
    print(f"Full splits: train={len(train_raw)}  val={len(val_raw)}  devtest={len(devtest_raw)}")

    TRAIN_N = os.environ.get("TRAIN_N")
    VAL_N = os.environ.get("VAL_N")
    if TRAIN_N is not None:
        train_raw = train_raw.select(range(min(int(TRAIN_N), len(train_raw))))
    if VAL_N is not None:
        val_raw = val_raw.select(range(min(int(VAL_N), len(val_raw))))
    print(f"Using: train={len(train_raw)}  val={len(val_raw)}"
          f"{'  (subset via TRAIN_N/VAL_N)' if (TRAIN_N or VAL_N) else ''}")

    CACHE_VERSION = f"v1_refcocoplus_maxlen512_right_train{len(train_raw)}_val{len(val_raw)}"
    CACHE_DIR = os.environ.get(
        "GRIT_REFCOCO_CACHE_DIR", os.path.expanduser("~/.cache/refcocoplus_grit"))
    TRAIN_CACHE = os.path.join(CACHE_DIR, f"tokenized_train_{CACHE_VERSION}")
    VAL_CACHE   = os.path.join(CACHE_DIR, f"tokenized_val_{CACHE_VERSION}")

    def load_or_tokenize(raw_dataset, cache_path, desc):
        if os.path.exists(cache_path):
            from datasets import load_from_disk
            ds = load_from_disk(cache_path)
            print(f"[{desc}] loaded {len(ds)} cached samples")
            return ds

        print(f"[{desc}] tokenizing {len(raw_dataset)} samples...")
        from functools import partial
        tok_fn = partial(tokenize_refcoco, config=config, processor=processor)

        # C1 FIX: tokenize_refcoco returns {"_skip": True} rather than None, which .map() can
        # handle (unlike None). Drop skipped rows with .filter() afterward.
        ds = raw_dataset.map(
            tok_fn,
            remove_columns=raw_dataset.column_names,
            num_proc=1,
            desc=f"Tokenizing {desc}",
            load_from_cache_file=False,
            writer_batch_size=50,  # small shards — pixel_values are large (see tokenize_local.py)
        )
        ds = ds.filter(lambda x: not x.get("_skip", False) and "input_ids" in x)
        os.makedirs(CACHE_DIR, exist_ok=True)
        ds.save_to_disk(cache_path)
        print(f"[{desc}] kept {len(ds)} / {len(raw_dataset)} samples -> {cache_path}")
        return ds

    tokenized_train = load_or_tokenize(train_raw, TRAIN_CACHE, "train")
    tokenized_val   = load_or_tokenize(val_raw,   VAL_CACHE,   "val")
    if len(tokenized_train) == 0:
        raise ValueError("No training samples survived tokenization — check the dataset/schema.")

    # -------------------------------------------------------------------
    # 3. Model — full precision (bf16), no quantization.
    # NOTE: this needs meaningfully more VRAM than the 4-bit path (~22GB for weights alone
    # vs. ~6-7GB in 4-bit NF4). Budget for an A100-40GB or larger; a 24GB card that worked
    # for the quantized version will very likely OOM here, especially with LoRA optimizer
    # states and activations on top.
    # -------------------------------------------------------------------
    print("Loading model onto GPU (bf16, no quantization)...")
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        device_map="auto", token=HF_TOKEN,
    )

    # H2 FIX: resolve target module NAMES explicitly instead of relying on target_modules name
    # matching to coincidentally avoid the vision tower. Language-only by default.
    leaf_names = tuple(config.lora_target_modules)  # q_proj, k_proj, ..., fc1, fc2
    apply_to_vision = bool(getattr(config, "apply_lora_to_vision", False))
    resolved_targets = []
    for name, _ in model.named_modules():
        if not name.endswith(leaf_names):
            continue
        is_vision = "vision_model" in name
        if is_vision and not apply_to_vision:
            continue
        resolved_targets.append(name)
    if not resolved_targets:
        raise RuntimeError("No target modules resolved — check lora_target_modules against "
                            "this model's actual module names (print model.named_modules()).")
    print(f"LoRA target modules resolved: {len(resolved_targets)} "
          f"({'language + vision' if apply_to_vision else 'language only'})")

    lora_config = LoraConfig(
        r=config.lora_rank, lora_alpha=config.lora_alpha,
        target_modules=resolved_targets, lora_dropout=config.lora_dropout,
        bias="none", task_type="CAUSAL_LM", inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    model.config.use_cache = False

    # -------------------------------------------------------------------
    # 4. Train
    # -------------------------------------------------------------------
    eff_batch = config.batch_size * config.gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(tokenized_train) / eff_batch)
    eval_steps = max(1, steps_per_epoch // 10)
    print(f"Steps/epoch: {steps_per_epoch} | eval every {eval_steps} steps")

    training_args = Seq2SeqTrainingArguments(
        output_dir=RUN_OUTPUT_DIR,
        per_device_train_batch_size=config.batch_size,
        # SPEED: eval carries no gradient/optimizer-state memory (only a forward pass),
        # so it can safely use a larger batch than training. Tune based on observed VRAM
        # headroom during eval — start conservative and raise if nvidia-smi shows room.
        per_device_eval_batch_size=int(os.environ.get("EVAL_BATCH_SIZE", 4)),
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        bf16=True,
        max_grad_norm=1.5,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # SPEED: safe to parallelize — tokenized_train/tokenized_val are pre-computed plain
        # tensors (pixel_values, input_ids, etc.), so worker processes only do collation,
        # never touch the model or GPU. pin_memory speeds up the host->device copy.
        dataloader_num_workers=int(os.environ.get("DATALOADER_WORKERS", 4)),
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none",
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=3,
        logging_steps=eval_steps,
        warmup_steps=50,
        optim="adamw_torch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        # C3 FIX: deliberately never enabled here — see module docstring. Real generation-based
        # IoU@0.5 is computed post-hoc by evaluate_refcoco_checkpoint.py.
        # Defense in depth alongside the prediction_step argmax fix in grit_llama.py: move
        # accumulated eval predictions to CPU every 10 eval batches instead of keeping the
        # whole eval set's predictions on GPU until compute_metrics runs.
        eval_accumulation_steps=10,
        predict_with_generate=False,
        # SEED: explicit so multi-seed comparison runs are reproducible and controllable.
        # Previously implicit (Trainer's default of 42) — meaning every run so far, GRIT or
        # LoRA-only, used the SAME seed. Seed-to-seed variance has never actually been
        # measured; this is what makes that experiment possible.
        seed=int(os.environ.get("SEED", 42)),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grit_manager = GRITManager(model, config, device)
    if len(grit_manager.optimized_modules) == 0:
        raise RuntimeError("GRITManager found 0 LoRA modules.")

    from functools import partial
    collate = partial(collate_fn, config=config, processor=processor)

    trainer_callbacks = [GritCallback(grit_manager)]
    if os.environ.get("DISABLE_EARLY_STOPPING", "0") != "1":
        patience = int(os.environ.get("EARLY_STOPPING_PATIENCE", 3))
        threshold = float(os.environ.get("EARLY_STOPPING_THRESHOLD", 0.01))
        trainer_callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=patience, early_stopping_threshold=threshold))
        print(f"Early stopping: patience={patience} threshold={threshold}")
    else:
        print("Early stopping: DISABLED (DISABLE_EARLY_STOPPING=1) — training will run the "
              "full step budget regardless of eval_loss plateauing. Useful for sanity "
              "checks where a flat/noisy eval_loss on few samples would otherwise halt "
              "training before GRIT's own activation thresholds are ever reached.")

    trainer = GritTrainer(
        processor=processor,
        grit_manager=grit_manager,
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=collate,
        compute_metrics=make_compute_metrics(processor),
        callbacks=trainer_callbacks,
    )

    # C2 FIX: only resume if a checkpoint genuinely exists.
    ckpt_dir = training_args.output_dir
    resume = None
    if os.path.isdir(ckpt_dir):
        candidates = [os.path.join(ckpt_dir, d) for d in os.listdir(ckpt_dir)
                      if d.startswith("checkpoint-")]
        if candidates:
            resume = max(candidates, key=lambda p: int(p.rsplit("-", 1)[-1]))
            print(f"Resuming from {resume}")

    try:
        train_start = time.time()
        trainer.train(resume_from_checkpoint=resume)
        train_wall_seconds = time.time() - train_start

        output_dir = f"{RUN_OUTPUT_DIR}_final_model"
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

        # Save full training history + run config for compare_runs.py. This is what makes
        # convergence-speed, cost-normalized-quality, and generalization-gap comparisons
        # possible after the fact, across any number of runs, without re-running anything.
        history_path = os.path.join(RUN_OUTPUT_DIR, "training_history.json")
        run_record = {
            "run_output_dir": RUN_OUTPUT_DIR,
            "train_wall_seconds": train_wall_seconds,
            "train_n": len(tokenized_train),
            "val_n": len(tokenized_val),
            "seed": training_args.seed,
            "ablation_config": {
                "disable_precondition": config.disable_precondition,
                "disable_reprojection": config.disable_reprojection,
                "disable_curvature_regularizer": config.disable_curvature_regularizer,
                "disable_reprojection_regularizer": config.disable_reprojection_regularizer,
                "track_covariance_for_analysis": config.track_covariance_for_analysis,
                "kfac_damping_ceiling": config.kfac_damping_ceiling,
            },
            "log_history": trainer.state.log_history,
        }
        os.makedirs(RUN_OUTPUT_DIR, exist_ok=True)
        with open(history_path, "w") as f:
            json.dump(run_record, f, indent=2, default=str)
        print(f"Training history saved to {history_path}")

        print_qualitative_examples(
            model, processor, devtest_raw, NUM_QUALITATIVE_EXAMPLES, output_dir)

        print("Now run evaluate_refcoco_checkpoint.py for the full IoU@0.5 benchmark on testA/testB.")
    except KeyboardInterrupt:
        interrupted_dir = f"{RUN_OUTPUT_DIR}_interrupted"
        model.save_pretrained(interrupted_dir)
        processor.save_pretrained(interrupted_dir)
        print(f"Interrupted — saved to {interrupted_dir}")
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
