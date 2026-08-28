"""
compare_runs.py
----------------
Compares any number of training runs (e.g. LoRA-only vs. GRIT, or different ablation
configs, or different seeds/dataset sizes) on dimensions that plot_heatmaps.py /
plot_spectra*.py don't touch at all: convergence speed, cost-normalized quality, and
generalization gap. Reads training_history.json, written by train_refcoco_grit.py at the
end of every run — no re-running or re-evaluation needed.

Usage:
  python compare_runs.py \
      --run ./results_lora_only          "LoRA only" \
      --run ./results_refcocoplus_grit_llama3_2_vision  "GRIT (full)"

  # Multiple seeds per config -> mean +/- std instead of a single curve
  python compare_runs.py \
      --run ./results_lora_seed42  "LoRA (seed 42)" \
      --run ./results_lora_seed43  "LoRA (seed 43)" \
      --run ./results_lora_seed44  "LoRA (seed 44)" \
      --group_prefix "LoRA"

Outputs (to --out_dir, default ./run_comparison):
  accuracy_vs_step.png       — eval_accuracy_at_0.5 over training, all runs overlaid
  loss_vs_step.png           — train loss AND eval_loss over training, all runs overlaid
  generalization_gap.png     — (eval_loss - train loss) over training, all runs overlaid
  steps_to_threshold.csv     — first step each run crosses each accuracy threshold
  cost_normalized_quality.csv — accuracy@0.5 per wall-clock second, at final eval
  summary.txt                — human-readable digest of everything above
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]


def load_history(run_dir: str) -> Dict:
    path = os.path.join(run_dir, "training_history.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"training_history.json not found at {path}. This is written by "
            "train_refcoco_grit.py at the end of training — make sure that run "
            "completed (or was at least interrupted after some training) with the "
            "updated script.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_series(log_history: List[Dict]) -> Dict[str, List[Tuple[float, float]]]:
    """
    Trainer's log_history mixes training-step rows (with 'loss') and eval rows (with
    'eval_loss', 'eval_accuracy_at_0.5', etc.) in one flat list, keyed loosely by 'step'
    and/or 'epoch'. Split into named (step, value) series.
    """
    series = defaultdict(list)
    for row in log_history:
        step = row.get("step")
        if step is None:
            continue
        for key in ("loss", "eval_loss", "eval_accuracy_at_0.5", "eval_mean_iou",
                    "eval_parse_failure_rate", "eval_runtime", "grad_norm"):
            if key in row and row[key] is not None:
                try:
                    series[key].append((float(step), float(row[key])))
                except (TypeError, ValueError):
                    pass
    return series


def steps_to_threshold(acc_series: List[Tuple[float, float]],
                        thresholds: List[float]) -> Dict[float, Optional[float]]:
    acc_series = sorted(acc_series, key=lambda t: t[0])
    result = {}
    for th in thresholds:
        hit = next((step for step, val in acc_series if val >= th), None)
        result[th] = hit
    return result


def cumulative_wall_seconds(record: Dict, acc_series: List[Tuple[float, float]]
                             ) -> List[Tuple[float, float]]:
    """
    Approximate wall-clock at each eval point by scaling total train_wall_seconds
    proportionally to step progress. Not exact (eval passes themselves cost time not
    evenly distributed), but good enough for a cost-normalized-quality comparison without
    needing per-step timestamps that Trainer doesn't log by default.
    """
    total_wall = record.get("train_wall_seconds")
    log_history = record.get("log_history", [])
    max_step = max((row.get("step", 0) for row in log_history), default=1) or 1
    if not total_wall:
        return []
    return [(step, (step / max_step) * total_wall) for step, _ in acc_series]


def group_and_average(runs: List[Dict], prefix: Optional[str]) -> List[Dict]:
    """
    Merge runs whose label starts with `prefix` into a single mean(+/-std) series —
    for comparing seed variance without cluttering the plot with N separate lines.
    Assumes matched runs share the same eval_steps schedule (true whenever they share
    TRAIN_N/VAL_N and only differ by SEED, since eval_steps is derived from dataset size,
    not from the seed).
    """
    if not prefix:
        return runs
    matched = [r for r in runs if r["label"].startswith(prefix)]
    unmatched = [r for r in runs if not r["label"].startswith(prefix)]
    if len(matched) < 2:
        print(f"[compare_runs] --group_prefix '{prefix}' matched fewer than 2 runs — "
              f"nothing to average, leaving as-is.")
        return runs

    def _avg_series(key: str):
        step_vals = defaultdict(list)
        for r in matched:
            for step, val in r["series"].get(key, []):
                step_vals[step].append(val)
        return step_vals

    acc_raw = _avg_series("eval_accuracy_at_0.5")
    merged_series = {
        "eval_accuracy_at_0.5": sorted((s, float(np.mean(v))) for s, v in acc_raw.items()),
        "loss": sorted((s, float(np.mean(v))) for s, v in _avg_series("loss").items()),
        "eval_loss": sorted((s, float(np.mean(v))) for s, v in _avg_series("eval_loss").items()),
    }
    acc_std = {s: float(np.std(v)) for s, v in acc_raw.items()}
    avg_wall = float(np.mean([r["record"].get("train_wall_seconds") or 0 for r in matched]))
    merged = {
        "dir": f"<{len(matched)} runs averaged: " + ", ".join(r["dir"] for r in matched) + ">",
        "label": f"{prefix} (mean, n={len(matched)})",
        "record": {
            "train_wall_seconds": avg_wall, "seed": "averaged",
            "train_n": matched[0]["record"].get("train_n"),
            "ablation_config": matched[0]["record"].get("ablation_config"),
        },
        "series": merged_series,
        "acc_std": acc_std,
    }
    print(f"[compare_runs] Averaged {len(matched)} runs matching prefix '{prefix}': "
          f"{[r['label'] for r in matched]}")
    return unmatched + [merged]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs=2, action="append", metavar=("RUN_DIR", "LABEL"),
                     required=True, help="Repeatable: a run directory and its display label.")
    ap.add_argument("--group_prefix", default=None,
                     help="If labels share this prefix (e.g. multiple seeds of the same "
                          "config), average them into one mean+/-std curve instead of "
                          "plotting each separately.")
    ap.add_argument("--out_dir", default="./run_comparison")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    runs = []
    for run_dir, label in args.run:
        record = load_history(run_dir)
        series = extract_series(record.get("log_history", []))
        runs.append({"dir": run_dir, "label": label, "record": record, "series": series})

    runs = group_and_average(runs, args.group_prefix)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(runs), 1)))

    # --- Plot 1: accuracy@0.5 vs step ---
    plt.figure(figsize=(8, 5))
    for i, r in enumerate(runs):
        acc = sorted(r["series"].get("eval_accuracy_at_0.5", []), key=lambda t: t[0])
        if not acc:
            print(f"[WARN] {r['label']}: no eval_accuracy_at_0.5 found — skipping in plot.")
            continue
        steps, vals = zip(*acc)
        plt.plot(steps, vals, marker="o", markersize=4, color=colors[i], label=r["label"])
        acc_std = r.get("acc_std")
        if acc_std:
            lo = [v - acc_std.get(s, 0) for s, v in acc]
            hi = [v + acc_std.get(s, 0) for s, v in acc]
            plt.fill_between(steps, lo, hi, color=colors[i], alpha=0.15)
    plt.xlabel("step"); plt.ylabel("accuracy@0.5 (teacher-forced proxy)")
    plt.title("Convergence speed comparison")
    plt.grid(alpha=0.3); plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "accuracy_vs_step.png"), dpi=args.dpi)
    plt.close()

    # --- Plot 2: train loss + eval loss vs step ---
    plt.figure(figsize=(8, 5))
    for i, r in enumerate(runs):
        tl = sorted(r["series"].get("loss", []), key=lambda t: t[0])
        el = sorted(r["series"].get("eval_loss", []), key=lambda t: t[0])
        if tl:
            steps, vals = zip(*tl)
            plt.plot(steps, vals, color=colors[i], linestyle="-", alpha=0.6,
                      label=f"{r['label']} (train)")
        if el:
            steps, vals = zip(*el)
            plt.plot(steps, vals, color=colors[i], linestyle="--", marker="o", markersize=3,
                      label=f"{r['label']} (eval)")
    plt.xlabel("step"); plt.ylabel("loss")
    plt.title("Train vs. eval loss")
    plt.grid(alpha=0.3); plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "loss_vs_step.png"), dpi=args.dpi)
    plt.close()

    # --- Plot 3: generalization gap (eval_loss - nearest train loss) vs step ---
    plt.figure(figsize=(8, 5))
    for i, r in enumerate(runs):
        tl = sorted(r["series"].get("loss", []), key=lambda t: t[0])
        el = sorted(r["series"].get("eval_loss", []), key=lambda t: t[0])
        if not tl or not el:
            continue
        tl_steps = np.array([s for s, _ in tl])
        tl_vals = np.array([v for _, v in tl])
        gap_steps, gap_vals = [], []
        for s, ev in el:
            nearest_idx = int(np.argmin(np.abs(tl_steps - s)))
            gap_steps.append(s)
            gap_vals.append(ev - tl_vals[nearest_idx])
        plt.plot(gap_steps, gap_vals, marker="o", markersize=4, color=colors[i], label=r["label"])
    plt.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    plt.xlabel("step"); plt.ylabel("eval_loss - train_loss (nearest step)")
    plt.title("Generalization gap (lower = less overfitting)")
    plt.grid(alpha=0.3); plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "generalization_gap.png"), dpi=args.dpi)
    plt.close()

    # --- CSV 1: steps-to-threshold ---
    csv_path = os.path.join(args.out_dir, "steps_to_threshold.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_label"] + [f"steps_to_acc>={th}" for th in THRESHOLDS])
        for r in runs:
            acc = r["series"].get("eval_accuracy_at_0.5", [])
            hits = steps_to_threshold(acc, THRESHOLDS)
            writer.writerow([r["label"]] + [hits[th] if hits[th] is not None else "never"
                                              for th in THRESHOLDS])
    print(f"[compare_runs] steps-to-threshold → {csv_path}")

    # --- CSV 2: cost-normalized quality ---
    csv_path2 = os.path.join(args.out_dir, "cost_normalized_quality.csv")
    with open(csv_path2, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_label", "train_wall_seconds", "final_accuracy_at_0.5",
                          "accuracy_per_1000_gpu_seconds"])
        for r in runs:
            wall = r["record"].get("train_wall_seconds")
            acc = sorted(r["series"].get("eval_accuracy_at_0.5", []), key=lambda t: t[0])
            final_acc = acc[-1][1] if acc else None
            rate = (final_acc / wall * 1000) if wall and final_acc is not None else None
            writer.writerow([r["label"], wall, final_acc, rate])
    print(f"[compare_runs] cost-normalized quality → {csv_path2}")

    # --- Human-readable summary ---
    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("RUN COMPARISON SUMMARY\n" + "=" * 60 + "\n\n")
        for r in runs:
            acc = sorted(r["series"].get("eval_accuracy_at_0.5", []), key=lambda t: t[0])
            el = sorted(r["series"].get("eval_loss", []), key=lambda t: t[0])
            wall = r["record"].get("train_wall_seconds")
            ablation = r["record"].get("ablation_config", {})
            f.write(f"{r['label']}  ({r['dir']})\n")
            f.write(f"  ablation config: {ablation}\n")
            f.write(f"  seed: {r['record'].get('seed')}  train_n: {r['record'].get('train_n')}\n")
            f.write(f"  wall clock: {wall:.1f}s\n" if wall else "  wall clock: unknown\n")
            if acc:
                f.write(f"  final accuracy@0.5: {acc[-1][1]:.3f} (step {acc[-1][0]:.0f})\n")
                f.write(f"  peak accuracy@0.5: {max(v for _, v in acc):.3f}\n")
            if el:
                f.write(f"  final eval_loss: {el[-1][1]:.4f}\n")
            f.write("\n")
    print(f"[compare_runs] summary → {summary_path}")
    print(f"\nAll outputs in {args.out_dir}/")


if __name__ == "__main__":
    main()
