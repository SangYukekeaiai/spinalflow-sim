#!/usr/bin/env python3
"""
Plot SRAM access counts across layers.

Usage:
    python plot_sram_accesses.py --csv stats/repo__model__sram_access.csv --output sram_access.png

The script draws one line per SRAM (input spine, filter, output queue) indexed by layer order.
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot SRAM access counts across layers.")
    parser.add_argument(
        "--csv",
        required=True,
        type=pathlib.Path,
        help="Path to the SRAM access CSV (…__sram_access.csv).",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="Optional output image path (PNG). Defaults to <csv>_access.png.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    required_cols = {
        "layer_id",
        "layer_name",
        "isb_accesses",
        "filter_accesses",
        "output_accesses",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns {missing} in {args.csv}")

    df = df.sort_values("layer_id").reset_index(drop=True)

    layers = df["layer_id"]
    layer_labels = df["layer_name"]

    output_path = args.output or args.csv.with_suffix("").with_name(args.csv.stem + "_access.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layers, df["isb_accesses"], marker="o", label="Input Spine")
    ax.plot(layers, df["filter_accesses"], marker="o", label="Filter Buffer")
    ax.plot(layers, df["output_accesses"], marker="o", label="Output Queue")

    ax.set_xlabel("Layer ID")
    ax.set_ylabel("Access Count")
    model = df["model"].iloc[0] if "model" in df.columns and not df.empty else "unknown_model"
    ax.set_title(f"{model} SRAM Accesses per Layer")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()

    if len(layer_labels.unique()) == len(layer_labels):
        ax.set_xticks(layers)
        ax.set_xticklabels(layer_labels, rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    print(f"[plot_sram_accesses] Wrote {output_path}")


if __name__ == "__main__":
    main()
