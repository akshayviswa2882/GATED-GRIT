"""
analyze_eval_results.py
-------------------------
Turns the eval_refcoco_results.json produced by evaluate_refcoco_checkpoint.py into a
reusable error-mode breakdown, instead of one-off ad-hoc analysis each time. Covers what
plot_heatmaps.py/plot_spectra*.py/compare_runs.py don't: what KINDS of errors a model
makes, not just how many.

Single-run mode:
  python analyze_eval_results.py --results ./eval_refcoco_results.json

Comparison mode (e.g. LoRA-only vs GRIT, same test set):
  python analyze_eval_results.py \
      --results ./eval_refcoco_results_lora.json --label "LoRA" \
      --compare_results ./eval_refcoco_results_grit.json --compare_label "GRIT"

Reports, per split (testA/testB) and pooled across both:
  - negation-expression accuracy vs. non-negation
  - near-full-image fallback rate ("model gave up and guessed everything")
  - short vs. long expression accuracy
  - worst-N and best-N examples
  - (comparison mode only) side-by-side deltas on all of the above
"""

import argparse
import json
import re
import statistics as stats
from typing import Dict, List, Optional, Tuple

NEGATION_WORDS = [" no ", " not ", "n't", " without "]
FULL_IMAGE_AREA_FRACTION = 0.85
GRID = 999


def parse_box(text: Optional[str]) -> Optional[Tuple[int, int, int, int]]:
    nums = re.findall(r"-?\d+", text or "")
    if len(nums) < 4:
        return None
    x1, y1, x2, y2 = (int(n) for n in nums[:4])
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def compute_iou(a: Optional[Tuple], b: Tuple) -> float:
    if a is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def analyze_split(samples: List[Dict]) -> Dict:
    scored = []
    for s in samples:
        pb = parse_box(s["prediction"])
        gt = tuple(s["gt_box_norm"])
        iou = compute_iou(pb, gt)
        expr = s["expression"]
        is_negation = any(w in f" {expr.lower()} " for w in NEGATION_WORDS)
        is_full_image = (pb is not None and
                          (pb[2] - pb[0]) * (pb[3] - pb[1]) > FULL_IMAGE_AREA_FRACTION * GRID * GRID)
        n_words = len(expr.split())
        scored.append({
            "expression": expr, "prediction": s["prediction"], "pred_box": pb,
            "gt_box": gt, "iou": iou, "is_negation": is_negation,
            "is_full_image": is_full_image, "n_words": n_words,
        })

    def _mean(vals):
        return sum(vals) / len(vals) if vals else None

    neg = [x for x in scored if x["is_negation"]]
    non_neg = [x for x in scored if not x["is_negation"]]
    full_img = [x for x in scored if x["is_full_image"]]
    short = [x for x in scored if x["n_words"] <= 2]
    long_ = [x for x in scored if x["n_words"] > 2]

    return {
        "n": len(scored),
        "mean_iou": _mean([x["iou"] for x in scored]),
        "accuracy_at_0.5": _mean([1.0 if x["iou"] >= 0.5 else 0.0 for x in scored]),
        "negation": {"n": len(neg), "mean_iou": _mean([x["iou"] for x in neg]),
                     "zero_iou_rate": _mean([1.0 if x["iou"] == 0 else 0.0 for x in neg])},
        "non_negation": {"n": len(non_neg), "mean_iou": _mean([x["iou"] for x in non_neg]),
                          "zero_iou_rate": _mean([1.0 if x["iou"] == 0 else 0.0 for x in non_neg])},
        "full_image_fallback": {"n": len(full_img), "rate": len(full_img) / len(scored) if scored else None,
                                 "mean_iou_when_it_happens": _mean([x["iou"] for x in full_img])},
        "short_expr": {"n": len(short), "mean_iou": _mean([x["iou"] for x in short])},
        "long_expr": {"n": len(long_), "mean_iou": _mean([x["iou"] for x in long_])},
        "worst_5": sorted(scored, key=lambda x: x["iou"])[:5],
        "best_3": sorted(scored, key=lambda x: -x["iou"])[:3],
    }


def print_split_report(name: str, report: Dict):
    print(f"\n--- {name} ---")
    print(f"  n={report['n']}  mean_iou={report['mean_iou']:.3f}  "
          f"accuracy@0.5={report['accuracy_at_0.5']:.3f}")
    neg, non_neg = report["negation"], report["non_negation"]
    if neg["n"]:
        print(f"  negation (n={neg['n']}): mean_iou={neg['mean_iou']:.3f}  "
              f"zero_iou_rate={neg['zero_iou_rate']:.2f}   vs.   "
              f"non-negation (n={non_neg['n']}): mean_iou={non_neg['mean_iou']:.3f}  "
              f"zero_iou_rate={non_neg['zero_iou_rate']:.2f}")
    fi = report["full_image_fallback"]
    print(f"  full-image fallback: {fi['n']}/{report['n']} ({fi['rate']:.1%})  "
          f"mean_iou when it happens={fi['mean_iou_when_it_happens']}")
    se, le = report["short_expr"], report["long_expr"]
    print(f"  short expr (<=2 words, n={se['n']}): mean_iou={se['mean_iou']}   vs.   "
          f"long expr (n={le['n']}): mean_iou={le['mean_iou']}")
    print(f"  worst 3:")
    for x in report["worst_5"][:3]:
        print(f"    {x['expression']!r}  pred={x['prediction']!r}  IoU={x['iou']:.3f}")


def load_results(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def analyze_file(path: str) -> Dict:
    data = load_results(path)
    return {split: analyze_split(data[split]["samples"]) for split in data
            if "samples" in data[split]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--label", default="run")
    ap.add_argument("--compare_results", default=None)
    ap.add_argument("--compare_label", default="comparison run")
    args = ap.parse_args()

    reports = analyze_file(args.results)
    print("=" * 70)
    print(f"ERROR-MODE ANALYSIS — {args.label} ({args.results})")
    print("=" * 70)
    for split, report in reports.items():
        print_split_report(split, report)

    if args.compare_results:
        reports_b = analyze_file(args.compare_results)
        print("\n" + "=" * 70)
        print(f"COMPARISON — {args.compare_label} ({args.compare_results})")
        print("=" * 70)
        for split, report in reports_b.items():
            print_split_report(split, report)

        print("\n" + "=" * 70)
        print(f"HEAD-TO-HEAD: {args.label} vs {args.compare_label}")
        print("=" * 70)
        for split in reports:
            if split not in reports_b:
                continue
            a, b = reports[split], reports_b[split]
            print(f"\n--- {split} ---")
            print(f"  accuracy@0.5:        {args.label}={a['accuracy_at_0.5']:.3f}   "
                  f"{args.compare_label}={b['accuracy_at_0.5']:.3f}   "
                  f"delta={b['accuracy_at_0.5'] - a['accuracy_at_0.5']:+.3f}")
            print(f"  mean_iou:            {args.label}={a['mean_iou']:.3f}   "
                  f"{args.compare_label}={b['mean_iou']:.3f}   "
                  f"delta={b['mean_iou'] - a['mean_iou']:+.3f}")
            print(f"  full-image fallback: {args.label}={a['full_image_fallback']['rate']:.1%}   "
                  f"{args.compare_label}={b['full_image_fallback']['rate']:.1%}")
            if a["negation"]["n"] and b["negation"]["n"]:
                print(f"  negation mean_iou:   {args.label}={a['negation']['mean_iou']:.3f}   "
                      f"{args.compare_label}={b['negation']['mean_iou']:.3f}")


if __name__ == "__main__":
    main()
