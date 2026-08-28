"""
plot_spectra_compare.py
-----------------------
Compare K-FAC eigenvalue spectra across multiple layers and module types.

Reads telemetry.json produced at end of GRIT training and generates:
  1. A grid of bar charts — one panel per selected module
  2. A single overlay line chart — all selected modules on one axis

Usage examples:

  # Compare q_proj across early / middle / late language layers
  python plot_spectra_compare.py \
      --run_dir ./results_vqa_grit_llama3_2_vision \
      --modules "language_model.layers.0.self_attn.q_proj" \
                "language_model.layers.15.self_attn.q_proj" \
                "language_model.layers.31.self_attn.q_proj"

  # Compare different module types in the same layer
  python plot_spectra_compare.py \
      --run_dir ./results_vqa_grit_llama3_2_vision \
      --modules "language_model.layers.15.self_attn.q_proj" \
                "language_model.layers.15.self_attn.v_proj" \
                "language_model.layers.15.mlp.gate_proj" \
                "language_model.layers.15.mlp.down_proj"

  # Vision vs language — same module type
  python plot_spectra_compare.py \
      --run_dir ./results_vqa_grit_llama3_2_vision \
      --modules "vision_model.transformer.layers.0.self_attn.q_proj" \
                "language_model.layers.0.self_attn.q_proj" \
      --side g

  # Auto-select: pick one module every N layers for a given module type
  python plot_spectra_compare.py \
      --run_dir ./results_vqa_grit_llama3_2_vision \
      --auto_type q_proj \
      --auto_stride 8

Output:
  <run_dir>/plots/spectra_grid.png    — grid of individual panels
  <run_dir>/plots/spectra_overlay.png — all modules overlaid on one chart
"""

import argparse
import json
import os
import math
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_telemetry(run_dir: str) -> Dict:
    path = os.path.join(run_dir, "telemetry.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"telemetry.json not found at {path}.\n"
            "Make sure training completed with the updated grit_llama.py "
            "which writes telemetry at the end of log_final_raw_ranks()."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_module(mods: List[Dict], substr: str) -> Optional[Dict]:
    """Return first module whose lora_name contains substr."""
    for m in mods:
        if substr in (m.get("lora_name") or ""):
            return m
    return None


def get_eigvals(mod: Dict, side: str, top_k: int) -> List[float]:
    key = "eigvals_a" if side == "a" else "eigvals_g"
    vals = mod.get(key) or []
    return [float(x) for x in vals[:top_k]]


def auto_select_modules(mods: List[Dict], module_type: str,
                        stride: int) -> List[str]:
    """
    Pick one module of the given type every `stride` layers,
    sorted by layer index ascending.
    """
    candidates = [
        m for m in mods
        if m.get("module_type") == module_type
        and m.get("layer") is not None
    ]
    candidates.sort(key=lambda m: int(m["layer"]))
    selected = candidates[::stride]
    return [m["lora_name"] for m in selected]


def short_label(lora_name: str) -> str:
    """
    Shorten a full lora_name to something that fits a chart legend.
    e.g. 'base_model.model.model.language_model.layers.15.self_attn.q_proj'
         -> 'lang.15.q_proj'
    """
    name = lora_name
    for prefix in [
        "base_model.model.model.",
        "base_model.model.",
        "model.model.",
        "model.",
    ]:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    name = name.replace("language_model.", "lang.")
    name = name.replace("vision_model.", "vis.")
    name = name.replace("global_transformer.", "glob.")
    name = name.replace("transformer.", "tr.")
    name = name.replace("self_attn.", "")
    name = name.replace("cross_attn.", "x_attn.")
    name = name.replace("mlp.", "")
    name = name.replace("layers.", "")
    return name


# ---------------------------------------------------------------------------
# Plot 1: Grid of individual panels
# ---------------------------------------------------------------------------

def plot_grid(selected_mods: List[Dict], side: str, top_k: int,
              out_path: str, dpi: int, run_dir: str, global_step: int):
    n = len(selected_mods)
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 3.2 * nrows),
        squeeze=False,
    )
    fig.suptitle(
        f"K-FAC eigenvalue spectra — {side.upper()}-side   "
        f"(step {global_step})",
        fontsize=13, y=1.01,
    )

    colors = cm.viridis(np.linspace(0.2, 0.85, n))

    for idx, mod in enumerate(selected_mods):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        eigs = get_eigvals(mod, side, top_k)
        label = short_label(mod.get("lora_name", f"module_{idx}"))

        if not eigs:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            ax.set_title(label, fontsize=9)
            ax.set_visible(True)
            continue

        xs = list(range(len(eigs)))
        ax.bar(xs, eigs, color=colors[idx], width=0.7, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(i) for i in xs], fontsize=8)
        ax.set_xlabel("eigenvalue index", fontsize=8)
        ax.set_ylabel("magnitude", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_title(label, fontsize=9)

        # Annotate effective rank
        k_a = mod.get("k_a", "?")
        k_b = mod.get("k_b", "?")
        eff  = mod.get("eff_rank_a" if side == "a" else "eff_rank_g", None)
        ann  = f"k={k_a if side=='a' else k_b}"
        if eff is not None:
            ann += f"  eff_rank={eff:.2f}"
        ax.annotate(ann, xy=(0.97, 0.95), xycoords="axes fraction",
                    ha="right", va="top", fontsize=7.5,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    # Hide unused panels
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_spectra_compare] grid  → {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Overlay line chart
# ---------------------------------------------------------------------------

def plot_overlay(selected_mods: List[Dict], side: str, top_k: int,
                 out_path: str, dpi: int, global_step: int):
    n = len(selected_mods)
    colors = cm.tab10(np.linspace(0, 1, max(n, 1)))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title(
        f"K-FAC eigenvalue spectra overlay — {side.upper()}-side   "
        f"(step {global_step})",
        fontsize=12,
    )

    any_data = False
    for idx, mod in enumerate(selected_mods):
        eigs = get_eigvals(mod, side, top_k)
        if not eigs:
            continue
        label = short_label(mod.get("lora_name", f"module_{idx}"))
        xs = list(range(len(eigs)))
        ax.plot(xs, eigs, marker="o", markersize=4,
                color=colors[idx], linewidth=1.6,
                label=label, zorder=3)
        any_data = True

    if not any_data:
        ax.text(0.5, 0.5, "no eigenvalue data found",
                ha="center", va="center", transform=ax.transAxes, color="gray")
    else:
        ax.set_xlabel("eigenvalue index", fontsize=10)
        ax.set_ylabel("magnitude", fontsize=10)
        ax.grid(alpha=0.3, zorder=0)
        ax.legend(fontsize=8, loc="upper right",
                  framealpha=0.85, ncol=max(1, n // 6))

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot_spectra_compare] overlay → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Compare K-FAC spectra across multiple layers/modules.")
    ap.add_argument(
        "--run_dir", required=True,
        help="Run directory containing telemetry.json")
    ap.add_argument(
        "--modules", nargs="*", default=None,
        help="Substrings to match module lora_names (space-separated). "
             "Mutually exclusive with --auto_type.")
    ap.add_argument(
        "--auto_type", default=None,
        help="Module type to auto-select (e.g. q_proj, v_proj, gate_proj). "
             "Picks one every --auto_stride layers.")
    ap.add_argument(
        "--auto_stride", type=int, default=8,
        help="Layer stride for auto-selection (default: 8).")
    ap.add_argument(
        "--side", choices=["a", "g"], default="a",
        help="Which covariance side to plot (A=input, G=gradient). Default: a")
    ap.add_argument(
        "--top_k", type=int, default=8,
        help="How many eigenvalues to show per module (default: 8).")
    ap.add_argument(
        "--out_dir", default=None,
        help="Output directory (default: <run_dir>/plots)")
    ap.add_argument(
        "--dpi", type=int, default=200)
    args = ap.parse_args()

    if args.modules and args.auto_type:
        ap.error("--modules and --auto_type are mutually exclusive.")
    if not args.modules and not args.auto_type:
        ap.error("Provide either --modules or --auto_type.")

    telem = load_telemetry(args.run_dir)
    mods: List[Dict] = list(telem.get("modules", []))
    global_step: int = int(telem.get("global_step", 0))

    if not mods:
        raise RuntimeError("telemetry.json has no modules — check training completed.")

    # Resolve which modules to plot
    if args.auto_type:
        lora_names = auto_select_modules(mods, args.auto_type, args.auto_stride)
        if not lora_names:
            available = sorted({m.get("module_type") for m in mods
                                 if m.get("module_type")})
            raise ValueError(
                f"No modules found with module_type='{args.auto_type}'.\n"
                f"Available types: {available}"
            )
        print(f"Auto-selected {len(lora_names)} '{args.auto_type}' modules "
              f"(stride={args.auto_stride}):")
        for n in lora_names:
            print(f"  {n}")
    else:
        lora_names = args.modules

    selected_mods = []
    for substr in lora_names:
        mod = find_module(mods, substr)
        if mod is None:
            print(f"[WARN] No module matched '{substr}' — skipping.")
        else:
            selected_mods.append(mod)

    if not selected_mods:
        raise RuntimeError(
            "None of the requested modules were found in telemetry.json.\n"
            "Tip: run with --auto_type q_proj to see available modules."
        )

    print(f"\nPlotting {len(selected_mods)} modules, "
          f"{args.side.upper()}-side, top-{args.top_k} eigenvalues.")

    out_dir = args.out_dir or os.path.join(args.run_dir, "plots")
    plot_grid(
        selected_mods, args.side, args.top_k,
        out_path=os.path.join(out_dir, "spectra_grid.png"),
        dpi=args.dpi, run_dir=args.run_dir, global_step=global_step,
    )
    plot_overlay(
        selected_mods, args.side, args.top_k,
        out_path=os.path.join(out_dir, "spectra_overlay.png"),
        dpi=args.dpi, global_step=global_step,
    )

    print(f"\nDone. Images written to {out_dir}/")
    print("  spectra_grid.png    — individual panel per module")
    print("  spectra_overlay.png — all modules overlaid for direct comparison")


if __name__ == "__main__":
    main()
