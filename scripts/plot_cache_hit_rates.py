#!/usr/bin/env python3
"""
Plot cache hit-rate comparisons across layers for multiple strategies.

The script scans the cross-layer comparison CSVs emitted by
`test_cache_vgg16_l6` (and similar sweep tools), then draws a grouped bar chart
where each strategy is rendered with a distinct color.
"""

import argparse
import csv
import math
import sys
from pathlib import Path
import re


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot cache hit rates per layer for multiple strategies.",
    )
    parser.add_argument(
        "csv_dir",
        help="Directory that contains the cross_layer_comparsion_stats CSV files.",
    )
    return parser.parse_args()


def load_hit_rates(csv_path: Path) -> dict[int, float]:
    hit_rates: dict[int, float] = {}
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if "layer" not in reader.fieldnames or "hit_rate" not in reader.fieldnames:
            raise ValueError(f"{csv_path} does not include required columns (layer, hit_rate).")
        for row in reader:
            layer = int(row["layer"])
            hit_rate = float(row["hit_rate"])
            hit_rates[layer] = hit_rate * 100.0  # convert to percentage
    return hit_rates


def strategy_name_from_filename(csv_path: Path) -> str:
    stem = csv_path.stem
    marker = "ways_"
    pos = stem.find(marker)
    if pos == -1:
        return stem
    return stem[pos + len(marker) :]


LRU_PATTERN = re.compile(r"^lru_(?:no_prefetch_buffer|prefetch(?:_buffer)?(?:_(\d+)(KB|B))?)$",
                         re.IGNORECASE)


def strategy_sort_key(name: str) -> tuple[int, float, str]:
    match = LRU_PATTERN.match(name)
    if match:
        size, unit = match.groups()
        if size is None:
            size_bytes = 0.0
        else:
            size_int = int(size)
            size_bytes = float(size_int * 1024 if unit and unit.upper() == "KB" else size_int)
        return (0, size_bytes, name)
    return (1, math.inf, name)


def main() -> int:
    args = parse_args()
    cross_dir = Path(args.csv_dir)
    if not cross_dir.exists():
        print(f"Cross-layer directory not found: {cross_dir}", file=sys.stderr)
        return 1

    csv_files = sorted(p for p in cross_dir.glob("*.csv") if p.is_file())
    if not csv_files:
        print(f"No CSV files found in {cross_dir}", file=sys.stderr)
        return 1

    strategies: dict[str, dict[int, float]] = {}
    for csv_file in csv_files:
        try:
            strategies[strategy_name_from_filename(csv_file)] = load_hit_rates(csv_file)
        except Exception as exc:
            print(f"Skipping {csv_file} ({exc})", file=sys.stderr)

    if not strategies:
        print("No valid strategy data found.", file=sys.stderr)
        return 1

    layers = sorted({layer for data in strategies.values() for layer in data.keys()})
    if not layers:
        print("No layer rows detected in CSV files.", file=sys.stderr)
        return 1

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"matplotlib is required to plot charts: {exc}", file=sys.stderr)
        return 1

    fig, ax = plt.subplots(figsize=(max(8.0, len(layers) * 0.6), 6.0))
    strategies_sorted = [name for name in sorted(strategies.keys(), key=strategy_sort_key)]
    bar_width = 0.8 / max(1, len(strategies_sorted))
    x_positions = range(len(layers))

    cmap = plt.get_cmap("tab10")
    for idx, strategy in enumerate(strategies_sorted):
        offsets = [x + idx * bar_width for x in x_positions]
        heights = [strategies[strategy].get(layer, float("nan")) for layer in layers]
        ax.bar(
            offsets,
            heights,
            width=bar_width,
            label=strategy.replace("_", " "),
            color=cmap(idx % cmap.N),
        )

    ax.set_xticks([x + bar_width * (len(strategies_sorted) - 1) / 2 for x in x_positions])
    ax.set_xticklabels(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Hit Rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Cache Hit Rate per Layer")
    legend = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])

    output_path = cross_dir / "cache_hit_rates.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Wrote plot to {output_path}")
    plt.close(fig)

    return 0


if __name__ == "__main__":
    sys.exit(main())
