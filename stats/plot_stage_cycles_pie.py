#!/usr/bin/env python3
"""
Generate a load/compute/store pie chart aggregated across all layers.

Usage:
    python plot_stage_cycles_pie.py --csv stats/repo__model__stage_cycles.csv --output stage_cycles_pie.png

If --output is omitted, the script writes next to the CSV with suffix "_pie.png".
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot aggregated stage cycle pie chart.")
    parser.add_argument(
        "--csv",
        required=True,
        type=pathlib.Path,
        help="Path to the stage cycles CSV (…__stage_cycles.csv).",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="Optional output image path (PNG). Defaults to <csv>_pie.png.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    for col in ("load_cycles", "compute_cycles", "store_cycles"):
        if col not in df.columns:
            raise SystemExit(f"Missing column '{col}' in {args.csv}")

    totals = df[["load_cycles", "compute_cycles", "store_cycles"]].sum()
    labels = ["Load", "Compute", "Store"]

    model = df["model"].iloc[0] if "model" in df.columns and not df.empty else "unknown_model"
    output_path = args.output or args.csv.with_suffix("").with_name(args.csv.stem + "_pie.png")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(totals, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.set_title(f"{model} Stage Cycles")
    ax.axis("equal")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"[plot_stage_cycles_pie] Wrote {output_path}")


if __name__ == "__main__":
    main()
