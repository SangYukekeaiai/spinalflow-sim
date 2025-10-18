#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render histogram figures for reuse distribution CSV files."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root directory to scan (defaults to current directory).",
    )
    parser.add_argument(
        "--pattern",
        default="reuse_distribution_*.csv",
        help="Glob pattern to match distribution CSVs (default: %(default)s).",
    )
    parser.add_argument(
        "--tail-share",
        type=float,
        default=0.99,
        help=(
            "Keep bins until cumulative share reaches this fraction and drop the remainder "
            "(default: %(default)s, set to 1.0 to disable trimming)."
        ),
    )
    return parser.parse_args()


def read_distribution(csv_path: Path):
    distances = []
    counts = []
    shares = []
    with csv_path.open("r", encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                distance = int(row["reuse_distance"])
                count = float(row["count"])
                share = float(row["share"])
            except (KeyError, ValueError):
                continue
            distances.append(distance)
            counts.append(count)
            shares.append(share)
    return distances, counts, shares


def trim_tail(distances, counts, shares, tail_share: float):
    if not distances or not counts or not shares:
        return distances, counts, shares, 0.0
    if not (0.0 < tail_share < 1.0):
        return distances, counts, shares, 0.0
    cumulative = 0.0
    cutoff_idx = len(distances) - 1
    for idx, share in enumerate(shares):
        cumulative += share
        if cumulative >= tail_share:
            cutoff_idx = idx
            break
    trimmed_distances = distances[: cutoff_idx + 1]
    trimmed_counts = counts[: cutoff_idx + 1]
    trimmed_shares = shares[: cutoff_idx + 1]
    dropped_share = sum(shares[cutoff_idx + 1 :])
    return trimmed_distances, trimmed_counts, trimmed_shares, dropped_share


def plot_histogram(csv_path: Path, distances, counts, shares, tail_share: float):
    if not distances or not counts:
        return
    distances, counts, shares, dropped = trim_tail(distances, counts, shares, tail_share)
    if not distances or not counts:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(distances, counts, color="#4C72B0", width=1.0, edgecolor="none")
    ax.set_title(csv_path.stem)
    ax.set_xlabel("Reuse distance")
    ax.set_ylabel("Access count")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    if dropped > 0:
        dropped_pct = dropped * 100.0
        ax.text(
            0.99,
            0.95,
            f"Tail dropped: {dropped_pct:.2f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
    fig.tight_layout()
    output_path = csv_path.with_suffix(".png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {output_path}")


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Root path {root} does not exist.", file=sys.stderr)
        return 1

    csv_files = sorted(root.rglob(args.pattern))
    if not csv_files:
        print("No distribution CSV files found.", file=sys.stderr)
        return 1

    for csv_path in csv_files:
        distances, counts, shares = read_distribution(csv_path)
        plot_histogram(csv_path, distances, counts, shares, args.tail_share)

    return 0


if __name__ == "__main__":
    sys.exit(main())
