"""
plot_spectra_vs_baseline.py
----------------------------
Head-to-head comparison of K-FAC/curvature structure between TWO separate runs — e.g. a
real GRIT run vs. a LoRA-only run with track_covariance_for_analysis=True (which measures
the same curvature statistics passively, without ever acting on them via preconditioning
or reprojection).

This is what plot_spectra.py / plot_spectra_compare.py / plot_heatmaps.py can't do: all
three only compare modules *within* a single telemetry.json. This script answers the
actual research question — "does GRIT converge to a structurally different (e.g. lower
effective rank, differently distributed) solution than plain LoRA would have, for the
SAME modules?" — by reading two telemetry.json files and comparing matched modules.

Outputs:
  1. A per-module grid of spectra: run A vs run B overlaid, one panel per selected module.
  2. A CSV summary: eff_rank_a/eff_rank_g and k_a/k_b for both runs, per module, plus the
     delta — this table is usually more useful than the plots for a written comparison.

Usage:
  # Compare a real GRIT run against a LoRA-shadow run, auto-selecting q_proj every 8 layers
  python plot_spectra_vs_baseline.py \
      --run_dir_a ./results_refcocoplus_grit_llama3_2_vision --label_a "GRIT" \
      --run_dir_b ./results_refcocoplus_lora_shadow          --label_b "LoRA (shadow)" \
      --auto_type q_proj --auto_stride 8

  # Compare specific modules explicitly
  python plot_spectra_vs_baseline.py \
      --run_dir_a ./results_refcocoplus_grit_llama3_2_vision --label_a "GRIT" \
      --run_dir_b ./results_refcocoplus_lora_shadow          --label_b "LoRA (shadow)" \
      --modules "language_model.layers.15.self_attn.q_proj" \
                "language_model.layers.15.mlp.gate_proj"

Output:
  <out_dir>/spectra_vs_baseline_grid.png
  <out_dir>/effective_rank_comparison.csv
"""

import argparse
import csv
import json
import math
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_telemetry(run_dir: str) -> Dict:
    path = os.path.join(run_dir, "telemetry.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"telemetry.json not found at {path}. "
            "Make sure that run finished training (or at least reached "
            "log_final_raw_ranks() at the end)."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_module(mods: List[Dict], substr: str) -> Optional[Dict]:
    for m in mods:
        if substr in (m.get("lora_name") or ""):
            return m
    return None


def auto_select_modules(mods: List[Dict], module_type: str, stride: int) -> List[str]:
    candidates = [m for m in mods if m.get("module_type") == module_type
                  and m.get("layer") is not None]
    candidates.sort(key=lambda m: int(m["layer"]))
    return [m["lora_name"] for m in candidates[::stride]]


def short_label(lora_name: str) -> str:
    name = lora_name
    for prefix in ["base_model.model.model.", "base_model.model.", "model.model.", "model."]:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    name = name.replace("language_model.", "lang.").replace("vision_model.", "vis.")
    name = name.replace("global_transformer.", "glob.").replace("transformer.", "tr.")
    name = name.replace("self_attn.", "").replace("cross_attn.", "x_attn.")
    name = name.replace("mlp.", "").replace("layers.", "")
    return name


def get_eigvals(mod: Dict, side: str, top_k: int) -> List[float]:
    key = "eigvals_a" if side == "a" else "eigvals_g"
    return [float(x) for x in (mod.get(key) or [])[:top_k]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir_a", required=True)
    ap.add_argument("--run_dir_b", required=True)
    ap.add_argument("--label_a", default="A")
    ap.add_argument("--label_b", default="B")
    ap.add_argument("--modules", nargs="*", default=None,
                     help="Substrings to match module lora_names. "
                          "Matched against run_dir_a's module list; the same lora_name is "
                          "then looked up in run_dir_b. Mutually exclusive with --auto_type.")
    ap.add_argument("--auto_type", default=None,
                     help="Module type to auto-select (e.g. q_proj). Picks one every "
                          "--auto_stride layers, from run_dir_a's module list.")
    ap.add_argument("--auto_stride", type=int, default=8)
    ap.add_argument("--side", choices=["a", "g"], default="a")
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--out_dir", default=None,
                     help="Default: <run_dir_a>/plots_vs_baseline")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    if args.modules and args.auto_type:
        ap.error("--modules and --auto_type are mutually exclusive.")
    if not args.modules and not args.auto_type:
        ap.error("Provide either --modules or --auto_type.")

    telem_a = load_telemetry(args.run_dir_a)
    telem_b = load_telemetry(args.run_dir_b)
    mods_a: List[Dict] = list(telem_a.get("modules", []))
    mods_b: List[Dict] = list(telem_b.get("modules", []))
    if not mods_a:
        raise RuntimeError(f"{args.run_dir_a}: telemetry.json has no modules.")
    if not mods_b:
        raise RuntimeError(f"{args.run_dir_b}: telemetry.json has no modules.")

    if args.auto_type:
        lora_names = auto_select_modules(mods_a, args.auto_type, args.auto_stride)
        if not lora_names:
            available = sorted({m.get("module_type") for m in mods_a if m.get("module_type")})
            raise ValueError(f"No modules of type '{args.auto_type}' in run_dir_a. "
                              f"Available: {available}")
    else:
        lora_names = args.modules

    # Resolve matched pairs, skipping anything missing from either run
    pairs = []
    for substr in lora_names:
        ma = find_module(mods_a, substr)
        mb = find_module(mods_b, substr)
        if ma is None:
            print(f"[WARN] '{substr}' not found in {args.label_a} — skipping.")
            continue
        if mb is None:
            print(f"[WARN] '{ma['lora_name']}' not found in {args.label_b} "
                  f"(architectures/module names must match) — skipping.")
            continue
        pairs.append((ma, mb))

    if not pairs:
        raise RuntimeError("No modules matched in both runs. Check that both runs used "
                            "the same base model/LoRA target_modules, and that run_dir_b "
                            "actually tracked covariance (track_covariance_for_analysis=True "
                            "if it was a LoRA-only baseline).")

    out_dir = args.out_dir or os.path.join(args.run_dir_a, "plots_vs_baseline")
    os.makedirs(out_dir, exist_ok=True)

    # --- Plot 1: grid of overlaid spectra, one panel per matched module ---
    n = len(pairs)
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.2 * nrows), squeeze=False)
    fig.suptitle(f"K-FAC spectra: {args.label_a} vs {args.label_b} — {args.side.upper()}-side",
                 fontsize=13, y=1.02)

    for idx, (ma, mb) in enumerate(pairs):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        label = short_label(ma.get("lora_name", f"module_{idx}"))

        eigs_a = get_eigvals(ma, args.side, args.top_k)
        eigs_b = get_eigvals(mb, args.side, args.top_k)
        width = 0.38
        if eigs_a:
            xs = np.arange(len(eigs_a))
            ax.bar(xs - width / 2, eigs_a, width=width, label=args.label_a,
                   color="#3b6fb6", zorder=3)
        if eigs_b:
            xs = np.arange(len(eigs_b))
            ax.bar(xs + width / 2, eigs_b, width=width, label=args.label_b,
                   color="#d97b3f", zorder=3)
        if not eigs_a and not eigs_b:
            ax.text(0.5, 0.5, "no eigenvalue data in either run", ha="center", va="center",
                    transform=ax.transAxes, color="gray")

        ax.set_title(label, fontsize=9)
        ax.set_xlabel("eigenvalue index", fontsize=8)
        ax.set_ylabel("magnitude", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        if idx == 0:
            ax.legend(fontsize=7)

        eff_key = "eff_rank_a" if args.side == "a" else "eff_rank_g"
        eff_a = ma.get(eff_key)
        eff_b = mb.get(eff_key)
        ann = (f"{args.label_a} eff_rank={eff_a:.2f}\n{args.label_b} eff_rank={eff_b:.2f}"
               if eff_a is not None and eff_b is not None else "")
        if ann:
            ax.annotate(ann, xy=(0.97, 0.95), xycoords="axes fraction", ha="right", va="top",
                        fontsize=7, bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75))

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    grid_path = os.path.join(out_dir, "spectra_vs_baseline_grid.png")
    plt.savefig(grid_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_spectra_vs_baseline] grid → {grid_path}")

    # --- Output 2: CSV summary — usually more useful than the plot for a written comparison ---
    csv_path = os.path.join(out_dir, "effective_rank_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "lora_name", "layer", "module_type",
            f"{args.label_a}_eff_rank_a", f"{args.label_b}_eff_rank_a", "delta_eff_rank_a",
            f"{args.label_a}_eff_rank_g", f"{args.label_b}_eff_rank_g", "delta_eff_rank_g",
            f"{args.label_a}_k_a", f"{args.label_b}_k_a",
            f"{args.label_a}_k_b", f"{args.label_b}_k_b",
        ])
        for ma, mb in pairs:
            era_a, era_b = ma.get("eff_rank_a"), mb.get("eff_rank_a")
            erg_a, erg_b = ma.get("eff_rank_g"), mb.get("eff_rank_g")
            d_era = (era_a - era_b) if era_a is not None and era_b is not None else ""
            d_erg = (erg_a - erg_b) if erg_a is not None and erg_b is not None else ""
            writer.writerow([
                ma.get("lora_name"), ma.get("layer"), ma.get("module_type"),
                era_a, era_b, d_era, erg_a, erg_b, d_erg,
                ma.get("k_a"), mb.get("k_a"), ma.get("k_b"), mb.get("k_b"),
            ])
    print(f"[plot_spectra_vs_baseline] summary → {csv_path}")
    print(f"\nCompared {len(pairs)} matched module(s) between {args.label_a} and {args.label_b}.")
    print("Note: k_a/k_b will show full LoRA rank for any run where reprojection never ran "
          "(e.g. a LoRA-only baseline) — eff_rank_a/eff_rank_g (computed from eigenvalue "
          "entropy, independent of reprojection) is the meaningful comparison in that case.")


if __name__ == "__main__":
    main()
