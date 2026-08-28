"""
Plot Fisher/K-FAC spectra from a run directory.

This reads `runs/*/telemetry.json` produced by `scripts/train.py` and plots the eigenvalue
spectrum (A-side or G-side) for a selected LoRA module.
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def _select_module(mods: List[Dict], substr: Optional[str]) -> Dict:
    if not mods:
        raise RuntimeError("telemetry.json has no modules.")
    if substr is None:
        return mods[0]
    for m in mods:
        if substr in (m.get("lora_name") or ""):
            return m
    raise ValueError(f"No module matched substring '{substr}'.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Path to a run directory (contains telemetry.json)")
    ap.add_argument("--module_substr", default=None, help="Substring to select a module (default: first module)")
    ap.add_argument("--side", choices=["a", "g"], default="a", help="Which spectrum to plot (A-side or G-side).")
    ap.add_argument("--top_k", type=int, default=16, help="How many eigenvalues to plot")
    ap.add_argument("--out", default=None, help="Output image path (default: <run_dir>/plots/spectrum_<side>.png)")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    telemetry_path = os.path.join(args.run_dir, "telemetry.json")
    if not os.path.exists(telemetry_path):
        raise FileNotFoundError(f"Missing telemetry.json: {telemetry_path}. Run training first.")

    with open(telemetry_path, "r", encoding="utf-8") as f:
        telem = json.load(f)

    mods: List[Dict] = list(telem.get("modules", []))
    m = _select_module(mods, args.module_substr)
    name = m.get("lora_name") or "<unknown>"
    eig_key = "eigvals_a" if args.side == "a" else "eigvals_g"
    eig = m.get(eig_key) or []
    if not eig:
        raise RuntimeError(f"No eigenvalues found under '{eig_key}' for module: {name}")
    eig = [float(x) for x in eig][: int(max(1, args.top_k))]

    out_path = args.out or os.path.join(args.run_dir, "plots", f"spectrum_{args.side}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(6, 3))
    plt.bar(list(range(len(eig))), eig)
    plt.title(f"Top-{len(eig)} eigenvalues ({args.side.upper()})\n{name}")
    plt.xlabel("index")
    plt.ylabel("eigenvalue")
    plt.tight_layout()
    plt.savefig(out_path, dpi=args.dpi)
    plt.close()
    print(f"[plot_spectra] wrote: {out_path}")


if __name__ == "__main__":
    main()


