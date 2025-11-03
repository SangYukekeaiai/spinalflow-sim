#!/usr/bin/env python3
"""Visualise per-layer tile reuse distributions as stacked bars."""

import argparse
import csv
import pathlib
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


LAYER_RE = re.compile(r"layer(\d+)\.csv$")


def _ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _parse_layer_id(path: pathlib.Path) -> int:
    match = LAYER_RE.match(path.stem)
    if match:
        return int(match.group(1))
    digits = re.findall(r"\d+", path.stem)
    return int(digits[0]) if digits else 10**9


def _load_layer(csv_path: pathlib.Path) -> Dict[str, float]:
    reuse_keys: List[int] = []
    counts: Dict[int, float] = {}
    spines: set[int] = set()
    tiles: set[int] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or len(header) <= 2:
            return {}
        reuse_keys = [int(col) for col in header[2:]]
        for row in reader:
            if len(row) < len(header):
                continue
            try:
                spine = int(row[0])
                tile = int(row[1])
            except Exception:
                continue
            spines.add(spine)
            tiles.add(tile)
            for key, val in zip(reuse_keys, row[2:]):
                try:
                    counts[key] = counts.get(key, 0.0) + float(val)
                except Exception:
                    continue
    if not spines or not tiles:
        return {}
    denom = float(len(spines) * len(tiles))
    return {f"reuse_{k}": counts.get(k, 0.0) / denom for k in reuse_keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot reuse-in-tile statistics across layers")
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing reuse_in_tiles_statistics layer*.csv files")
    parser.add_argument("--output",
                        help="Output PDF path (default: <input-dir>/reuse_in_tile_distribution.pdf)")
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    files = sorted(input_dir.glob("layer*.csv"), key=_parse_layer_id)
    if not files:
        raise SystemExit("No layer*.csv files found.")

    layer_labels: List[str] = []
    layer_data: List[Dict[str, float]] = []
    reuse_keys: set[str] = set()
    totals: List[float] = []

    for csv_path in files:
        data = _load_layer(csv_path)
        if not data:
            continue
        layer_labels.append(csv_path.stem)
        layer_data.append(data)
        reuse_keys.update(data.keys())
        totals.append(sum(data.values()))

    if not layer_data:
        raise SystemExit("No valid reuse-in-tile data found.")

    reuse_keys_sorted = sorted(reuse_keys, key=lambda k: int(k.split('_')[1]))
    x = np.arange(len(layer_data))
    width = 0.6

    fig, ax = plt.subplots(figsize=(12, max(4, len(layer_data) * 0.4)))
    bottom = np.zeros(len(layer_data))

    for key in reuse_keys_sorted:
        heights = np.array([data.get(key, 0.0) for data in layer_data])
        bars = ax.bar(x, heights, width, bottom=bottom, label=f"reuse {key.split('_')[1]}")
        for bar, height, total in zip(bars, heights, totals):
            if height <= 0 or total <= 0:
                continue
            ratio = height / total
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + height / 2,
                    f"{ratio:.0%}",
                    ha="center", va="center", fontsize=8, color="white")
        bottom += heights

    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels, rotation=45, ha="right")
    ax.set_ylabel("Average unique addresses per tile")
    ax.set_xlabel("Layer")
    ax.set_title("Reuse-in-tile distribution by layer")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, axis="y", alpha=0.2, linestyle=":")
    fig.tight_layout()

    output_path = pathlib.Path(args.output) if args.output else (
        input_dir / "reuse_in_tile_distribution.pdf"
    )
    _ensure_dir(output_path.parent)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    main()
