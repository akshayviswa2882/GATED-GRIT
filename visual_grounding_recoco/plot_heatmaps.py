"""
Generate rank/allocation heatmaps from a run directory.

This reads `runs/*/telemetry.json` produced by `scripts/train.py` and plots a heatmap of
final effective ranks (k) across layers and module types.
"""

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


DEFAULT_ORDER = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Path to a run directory (contains telemetry.json)")
    ap.add_argument("--which", choices=["k_a", "k_b"], default="k_b", help="Which k to plot (A-side or B-side).")
    ap.add_argument("--out", default=None, help="Output image path (default: <run_dir>/plots/rank_heatmap.png)")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    telemetry_path = os.path.join(args.run_dir, "telemetry.json")
    if not os.path.exists(telemetry_path):
        raise FileNotFoundError(f"Missing telemetry.json: {telemetry_path}. Run training first.")

    with open(telemetry_path, "r", encoding="utf-8") as f:
        telem = json.load(f)

    mods: List[Dict] = list(telem.get("modules", []))
    rows = [m for m in mods if isinstance(m, dict) and m.get("layer") is not None and m.get("module_type") is not None]
    if not rows:
        raise RuntimeError("No layer/module_type entries found in telemetry.json (nothing to plot).")

    max_layer = max(int(r["layer"]) for r in rows)
    layers = list(range(max_layer + 1))
    module_types = [t for t in DEFAULT_ORDER if any(r.get("module_type") == t for r in rows)]
    # Fallback: include any other module types encountered
    extras = sorted({r.get("module_type") for r in rows if r.get("module_type") not in set(DEFAULT_ORDER)})
    module_types.extend(extras)

    mat = np.full((len(layers), len(module_types)), np.nan, dtype=np.float32)
    idx_type = {t: j for j, t in enumerate(module_types)}
    for r in rows:
        i = int(r["layer"])
        t = r.get("module_type")
        j = idx_type.get(t)
        if j is None:
            continue
        v = r.get(args.which)
        try:
            mat[i, j] = float(v) if v is not None else np.nan
        except Exception:
            mat[i, j] = np.nan

    out_path = args.out or os.path.join(args.run_dir, "plots", "rank_heatmap.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(max(6.0, 0.9 * len(module_types)), max(4.0, 0.25 * len(layers))))
    sns.heatmap(mat, xticklabels=module_types, yticklabels=layers, cmap="viridis", cbar=True)
    plt.title(f"GRIT final rank heatmap ({args.which})")
    plt.xlabel("Module type")
    plt.ylabel("Layer")
    plt.tight_layout()
    plt.savefig(out_path, dpi=args.dpi)
    plt.close()
    print(f"[plot_heatmaps] wrote: {out_path}")


if __name__ == "__main__":
    main()


