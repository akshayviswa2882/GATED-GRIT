# -*- coding: utf-8 -*-
"""
metrics_refcoco.py
-------------------
IoU-based metrics for RefCOCO+ referring-expression comprehension.

Two distinct things live here, and it's important not to conflate them (this is exactly the
distinction that was blurred in the original VQA project's in-training eval — see the code
review, finding C3):

  1. `make_compute_metrics()` — a CHEAP, SAFE proxy used during training via Trainer's normal
     teacher-forced eval. It decodes the model's argmax predictions at the answer positions
     (from ordinary forward-pass logits, never from `.generate()`), so it cannot leak the
     ground truth into a generation prompt — there is no generation happening here at all.
     It's directionally useful for early-stopping-style monitoring but is NOT real generation
     accuracy, because teacher forcing gives the model the correct previous tokens at every step.

  2. `iou_accuracy()` — the real metric, meant to be run on genuinely free-run generations
     (see evaluate_refcoco_checkpoint.py, which builds a fresh prompt-only input and calls
     `model.generate()` correctly). This is what should be reported as "RefCOCO+ accuracy@0.5".
"""

import numpy as np
from transformers import EvalPrediction

from data_utils_RefCOCOplus_Llama import parse_box, GRID


def compute_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def iou_accuracy(pred_texts, gt_boxes, threshold: float = 0.5):
    """
    pred_texts: list[str] — raw generated (or decoded) text per sample.
    gt_boxes:   list[(x1,y1,x2,y2)] — ground-truth boxes on the same 0..GRID grid.
    Returns dict with accuracy@threshold, mean IoU (parse failures count as IoU=0 for both),
    and parse_failure_rate reported separately so a formatting problem isn't mistaken for a
    localization problem.
    """
    ious, hits, failures = [], 0, 0
    for text, gt in zip(pred_texts, gt_boxes):
        pred = parse_box(text)
        if pred is None:
            failures += 1
            ious.append(0.0)
            continue
        iou = compute_iou(pred, gt)
        ious.append(iou)
        if iou >= threshold:
            hits += 1
    n = max(1, len(pred_texts))
    return {
        f"accuracy_at_{threshold}": hits / n,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "parse_failure_rate": failures / n,
        "num_samples": len(pred_texts),
    }


def make_compute_metrics(processor):
    """Teacher-forced proxy metric factory for Trainer's `compute_metrics=`. See module docstring."""

    def extract_answer_span(text: str) -> str:
        text = text.replace("<|begin_of_text|>", "").replace("<|end_of_text|>", "")
        text = text.replace("<|eot_id|>", "").replace("<|finetune_right_pad_id|>", "")
        if "assistant" in text.lower():
            parts = text.split("assistant")
            if len(parts) > 1:
                text = parts[-1]
        return text.strip()

    def compute_refcoco_metrics(p: EvalPrediction):
        def to_int_ids(arr):
            if isinstance(arr, (list, tuple)):
                arr = arr[0]
            arr = np.asarray(arr)
            if arr.ndim == 3:
                arr = arr.argmax(-1)
            return arr.astype(np.int64)

        pad_id = processor.tokenizer.pad_token_id
        ignore_index = -100

        pred_ids = to_int_ids(p.predictions)
        vocab_size = processor.tokenizer.vocab_size
        if vocab_size is not None:
            bad = (pred_ids < 0) | (pred_ids >= vocab_size)
            pred_ids[bad] = pad_id if pad_id is not None else 0

        labels = np.asarray(p.label_ids, dtype=np.int64)

        pred_texts, gt_boxes = [], []
        for pred_row, label_row in zip(pred_ids, labels):
            valid = (label_row != ignore_index) & (label_row != (pad_id if pad_id is not None else -1))
            if valid.sum() == 0:
                continue
            # Teacher-forced: predicted token at position i is the model's guess for
            # position i+1. Align predictions to the same span the labels occupy.
            answer_pred_ids = pred_row[valid]
            answer_gt_ids = label_row[valid]
            try:
                pred_text = extract_answer_span(
                    processor.tokenizer.decode(answer_pred_ids, skip_special_tokens=True))
                gt_text = extract_answer_span(
                    processor.tokenizer.decode(answer_gt_ids, skip_special_tokens=True))
            except Exception:
                continue
            gt_box = parse_box(gt_text)
            if gt_box is None:
                continue  # can't score without a valid ground-truth box
            pred_texts.append(pred_text)
            gt_boxes.append(gt_box)

        if not pred_texts:
            return {"accuracy_at_0.5": 0.0, "mean_iou": 0.0, "parse_failure_rate": 1.0}

        stats = iou_accuracy(pred_texts, gt_boxes, threshold=0.5)
        print(f"\n[RefCOCO+ proxy metrics — teacher-forced, NOT real generation] {stats}\n")
        return stats

    return compute_refcoco_metrics
