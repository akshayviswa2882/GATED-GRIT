# -*- coding: utf-8 -*-
"""
data_utils_RefCOCOplus_Llama.py
--------------------------------
Data layer for porting GRIT (grit_llama.py, unchanged) to RefCOCO+ referring-expression
comprehension (REC): given an image + a referring expression, output a bounding box.

Design choices relative to the LLaVA-VQA version of this project (see code review):
  - Images come pre-embedded in the HF dataset (lmms-lab/RefCOCOplus) — there is no COCO
    filename-lookup / placeholder-image problem here at all. Every sample trains on a real image.
  - Boxes are represented as TEXT so the rest of the pipeline (Llama-3.2-Vision + LoRA + GRIT)
    needs zero architectural changes: "[x1,y1,x2,y2]" with integers on a fixed 0-999 grid,
    normalized per-axis by that sample's own image width/height. IoU computed on this grid is
    mathematically identical to pixel-space IoU (both boxes get the same per-axis scale factor),
    so no denormalization is needed downstream.
  - Each object has multiple human-written referring expressions (dataset's `answer` list).
    Duplicating a sample once per expression would multiply pixel_values storage N times (the
    exact PyArrow-2GB problem tokenize_local.py works around in the VQA project) — so instead
    we deterministically pick ONE expression per object at tokenize time, keyed off a stable hash
    of the sample id, so it's not systematically biased toward "the first" expression the way
    the original project's _extract_llava_qa is.
"""

import os
import re
import hashlib
import torch
from typing import Optional

from datasets import load_dataset


# -------------------------------------------------------------------------
# Dataset loading
# -------------------------------------------------------------------------

def load_refcocoplus_pool(seed: int = 42):
    """
    lmms-lab/RefCOCOplus has splits: val (3.81k), testA (1.98k), testB (1.8k) — no `train`.
    We treat `val` as our working pool and carve train/val/dev-test out of it ourselves
    (same seeded-shuffle-and-slice pattern as the original VQA project), and keep testA/testB
    untouched as the real, published-comparable RefCOCO+ benchmark splits for final evaluation.
    """
    pool = load_dataset("lmms-lab/RefCOCOplus", split="val").shuffle(seed=seed)
    n = len(pool)
    # ~2/3 train, ~1/6 val, remainder dev-test (mirrors the 2000/250/250-style split ratio)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    train_dataset = pool.select(range(0, n_train))
    val_dataset   = pool.select(range(n_train, n_train + n_val))
    devtest_dataset = pool.select(range(n_train + n_val, n))
    return train_dataset, val_dataset, devtest_dataset


def load_refcocoplus_benchmark_splits():
    """The official held-out RefCOCO+ test splits — use these for final reported numbers."""
    testA = load_dataset("lmms-lab/RefCOCOplus", split="testA")
    testB = load_dataset("lmms-lab/RefCOCOplus", split="testB")
    return testA, testB


# -------------------------------------------------------------------------
# Box <-> normalized-grid conversion
# -------------------------------------------------------------------------

GRID = 999  # coordinates are integers in [0, GRID]

def box_xywh_to_norm(box_xywh, img_w: int, img_h: int, grid: int = GRID):
    """[x, y, w, h] absolute pixels -> (x1, y1, x2, y2) ints on a 0..grid grid."""
    x, y, w, h = box_xywh
    x1, y1, x2, y2 = x, y, x + w, y + h
    if img_w <= 0 or img_h <= 0:
        return 0, 0, 0, 0

    def _norm(v, size):
        v = max(0.0, min(1.0, float(v) / float(size)))
        return int(round(v * grid))

    x1n, x2n = _norm(x1, img_w), _norm(x2, img_w)
    y1n, y2n = _norm(y1, img_h), _norm(y2, img_h)
    x1n, x2n = min(x1n, x2n), max(x1n, x2n)
    y1n, y2n = min(y1n, y2n), max(y1n, y2n)
    return x1n, y1n, x2n, y2n


def format_box(box_norm) -> str:
    x1, y1, x2, y2 = box_norm
    return f"[{x1},{y1},{x2},{y2}]"


_BOX_RE = re.compile(r"-?\d+")

def parse_box(text: str, grid: int = GRID):
    """
    Extract the first 4 integers found in generated text and clip them to the grid.
    Returns None if fewer than 4 numbers are present (a genuine parse failure, kept
    separate from "wrong box" downstream so the two failure modes aren't conflated).
    """
    if not text:
        return None
    nums = _BOX_RE.findall(text)
    if len(nums) < 4:
        return None
    x1, y1, x2, y2 = (int(n) for n in nums[:4])
    x1 = max(0, min(grid, x1)); x2 = max(0, min(grid, x2))
    y1 = max(0, min(grid, y1)); y2 = max(0, min(grid, y2))
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return x1, y1, x2, y2


# -------------------------------------------------------------------------
# Two-pass tokenization (mirrors tokenize_vqa's label-masking method exactly)
# -------------------------------------------------------------------------

SYSTEM_MESSAGE = (
    "You are a visual grounding assistant. Given an image and a referring expression, "
    "respond with ONLY the bounding box of the referred object, formatted exactly as "
    "[x1,y1,x2,y2], using integers from 0 to 999 that represent the box position scaled "
    "to the image's width and height, with the origin at the top-left corner."
)


def _pick_expression(example) -> Optional[str]:
    exprs = example.get("answer") or []
    exprs = [e for e in exprs if isinstance(e, str) and e.strip()]
    if not exprs:
        return None
    qid = str(example.get("question_id", example.get("file_name", "")))
    idx = int(hashlib.sha1(qid.encode("utf-8")).hexdigest(), 16) % len(exprs)
    return exprs[idx].strip()


def tokenize_refcoco(example, config, processor):
    """
    Returns a dict of tensors on success. On any failure returns {"_skip": True} —
    NEVER returns None (Dataset.map() cannot handle a function that returns None for
    some rows and a dict for others; filter on "_skip" afterward instead — see
    train_refcoco_grit.py's load_or_tokenize()).
    """
    try:
        image = example.get("image", None)
        bbox = example.get("bbox", None)
        if image is None or bbox is None or len(bbox) != 4:
            return {"_skip": True}

        expression = _pick_expression(example)
        if not expression:
            return {"_skip": True}

        if image.mode != "RGB":
            image = image.convert("RGB")
        img_w, img_h = image.size

        box_norm = box_xywh_to_norm(bbox, img_w, img_h)
        if box_norm[2] <= box_norm[0] or box_norm[3] <= box_norm[1]:
            return {"_skip": True}  # degenerate zero-area box
        answer_text = format_box(box_norm)

        user_text = f'Referring expression: "{expression}"\nOutput the bounding box.'

        system_message = {"role": "system", "content": SYSTEM_MESSAGE}
        messages_with_answer = [
            system_message,
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
            {"role": "assistant", "content": [{"type": "text", "text": answer_text}]},
        ]
        messages_prompt_only = [
            system_message,
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
        ]

        input_text_full = processor.apply_chat_template(
            messages_with_answer, add_generation_prompt=False, tokenize=False)
        input_text_prompt = processor.apply_chat_template(
            messages_prompt_only, add_generation_prompt=True, tokenize=False)

        max_length = getattr(config, "max_length", 512)
        enc_full = processor(images=[image], text=[input_text_full], padding="max_length",
                              max_length=max_length, truncation=True, return_tensors="pt")
        enc_prompt = processor(images=[image], text=[input_text_prompt], padding="max_length",
                                max_length=max_length, truncation=True, return_tensors="pt")

        input_ids = enc_full["input_ids"].squeeze(0)
        attention_mask = enc_full["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        ignore_index = getattr(config, "ignore_index", -100)
        pad_id = processor.tokenizer.pad_token_id

        # Same BUG-5-style fix as tokenize_vqa: prompt_len via attention-mask sum, right-padding
        # required (enforced by the caller, mirrored below in train_refcoco_grit.py).
        prompt_len = int(enc_prompt["attention_mask"].squeeze(0).sum().item())
        labels[:prompt_len] = ignore_index
        if pad_id is not None:
            labels[input_ids == pad_id] = ignore_index

        if (labels != ignore_index).sum() == 0:
            return {"_skip": True}

        out = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        for k in ("pixel_values", "aspect_ratio_ids", "aspect_ratio_mask",
                  "cross_attention_mask", "image_sizes"):
            if k in enc_full:
                v = enc_full[k].squeeze(0)
                out[k] = v.numpy().astype("float32") if k == "pixel_values" else v

        # Prompt-only fields for real generation (used only by evaluate_refcoco_checkpoint.py
        # and, if you ever re-enable it, the training-time generation eval).
        out["eval_input_ids"] = enc_prompt["input_ids"].squeeze(0)
        out["eval_attention_mask"] = enc_prompt["attention_mask"].squeeze(0)
        if "cross_attention_mask" in enc_prompt:
            out["eval_cross_attention_mask"] = enc_prompt["cross_attention_mask"].squeeze(0)

        # Kept for eval-time IoU scoring — NOT used as a model input.
        out["gt_box_norm"] = list(box_norm)

        return out

    except Exception as e:
        print(f"[tokenize_refcoco] error: {e}")
        return {"_skip": True}


# -------------------------------------------------------------------------
# Collator — generic, task-agnostic (same shape contract as the VQA project's collate_fn,
# plus the eval_* keys so a future generation-eval path has real data to work with).
# -------------------------------------------------------------------------

def collate_fn(batch, config, processor):
    batch = [b for b in batch if b is not None and isinstance(b, dict) and 'input_ids' in b]
    if len(batch) == 0:
        return _create_dummy_batch(config, processor)

    output = {}
    text_keys = ['input_ids', 'attention_mask', 'labels',
                 'eval_input_ids', 'eval_attention_mask']
    for key in text_keys:
        if key not in batch[0]:
            continue
        tensor_list = []
        for b in batch:
            item = b[key]
            if not isinstance(item, torch.Tensor):
                item = torch.tensor(item, dtype=torch.long)
            tensor_list.append(item)
        if key == 'labels':
            pad_val = getattr(config, "ignore_index", -100)
        elif key in ('input_ids', 'eval_input_ids'):
            pad_val = processor.tokenizer.pad_token_id
        else:
            pad_val = 0
        output[key] = torch.nn.utils.rnn.pad_sequence(
            tensor_list, batch_first=True, padding_value=pad_val)

    vision_keys = ["pixel_values", "image_sizes", "aspect_ratio_ids",
                   "aspect_ratio_mask", "cross_attention_mask", "eval_cross_attention_mask"]
    for key in vision_keys:
        if key in batch[0] and batch[0][key] is not None:
            items = []
            for b in batch:
                val = b.get(key)
                if val is None:
                    continue
                if not isinstance(val, torch.Tensor):
                    val = torch.tensor(val, dtype=torch.float32 if key == "pixel_values" else torch.long)
                items.append(val)
            if items:
                try:
                    output[key] = torch.stack(items, dim=0)
                except Exception as e:
                    print(f"Error stacking {key}: {e}")

    # Ground-truth boxes for eval-time IoU — kept as a plain Python list, not a tensor
    # (variable use downstream, never fed to the model).
    if "gt_box_norm" in batch[0]:
        output["gt_box_norm"] = [b.get("gt_box_norm") for b in batch]

    return output


def _create_dummy_batch(config, processor):
    print("WARNING: Creating dummy batch")
    max_len = getattr(config, "max_length", 512)
    return {
        'input_ids': torch.zeros((1, max_len), dtype=torch.long),
        'attention_mask': torch.zeros((1, max_len), dtype=torch.long),
        'labels': torch.full((1, max_len), getattr(config, "ignore_index", -100), dtype=torch.long),
    }
