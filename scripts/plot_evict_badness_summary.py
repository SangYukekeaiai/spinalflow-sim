#!/usr/bin/env python3
"""
Visualise aggregated bad-eviction counts per output spine/tile.

CSV format (emitted as evict_badness_summary.csv):
  output spine id,tile id,bad eviction total,bad eviction 1 times,...

Each page of the resulting PDF contains 3x3 subplots. Every subplot
corresponds to an output spine, the x-axis enumerates tile IDs, and stacked
bars show how many lines experienced a bad eviction with the indicated delay.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot aggregated bad eviction statistics.")
    parser.add_argument("csv_path", help="Path to evict_badness_summary.csv")
    return parser.parse_args()


def _normalise_spine_token(token: str) -> str:
    token = token.strip()
    if token.startswith("(") and token.endswith(")"):
        return token
    match = re.match(r"^(\d+)_?(\d+)?$", token)
    if match and match.group(2) is not None:
        return f"({int(match.group(1))}, {int(match.group(2))})"
    return token


def load_summary(csv_path: Path):
    import csv

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV missing header")

        buckets = [name for name in reader.fieldnames if name.lower().startswith("bad eviction ")]
        buckets.sort(key=lambda s: int("".join(filter(str.isdigit, s)) or 0))

        data: Dict[str, Dict[int, Dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
        for row in reader:
            try:
                spine = _normalise_spine_token(row["output spine id"])
                tile = int(row["tile id"])
                total = int(row.get("bad eviction total", "0") or 0)
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Malformed row: {row}") from exc

            entry: Dict[str, int] = {"total": total}
            for bucket in buckets:
                entry[bucket] = int(row.get(bucket, "0") or 0)
            data[spine][tile] = entry

    return buckets, data


def _spine_sort_key(label: str):
    if label.startswith("(") and label.endswith(")"):
        try:
            parts = [int(x.strip()) for x in label.strip("()").split(",")]
            if len(parts) == 2:
                return (0, parts[0], parts[1])
        except ValueError:
            pass
    try:
        return (1, int(label))
    except ValueError:
        return (2, label)


def make_plots(csv_path: Path, buckets: List[str], data) -> None:
    if not data:
        print("No data to plot.")
        return

    spines = sorted(data.keys(), key=_spine_sort_key)
    colors = ["#4C78A8", "#F58518", "#E4572E", "#72B7B2", "#B279A2", "#FF9DA7"]
    labels = [bucket.replace("_", " ").title() for bucket in buckets]
    legend_patches = [
        Patch(facecolor=colors[idx % len(colors)], edgecolor="black", label=labels[idx])
        for idx in range(len(buckets))
    ]

    per_page = 8
    rows, cols = 3, 3
    pdf_path = csv_path.parent / "small_multiple_eviction_badness_summary.pdf"

    with PdfPages(pdf_path) as pdf:
        for start in range(0, len(spines), per_page):
            page_spines = spines[start : start + per_page]
            nrows = min(rows, math.ceil(len(page_spines) / cols))
            fig, axes = plt.subplots(
                nrows,
                cols,
                figsize=(cols * 3.5, nrows * 3.0),
                squeeze=False,
            )

            for ax in axes.flatten():
                ax.axis("off")

            for idx, spine in enumerate(page_spines):
                r, c = divmod(idx, cols)
                ax = axes[r][c]
                ax.axis("on")
                tile_map = data[spine]
                tiles = sorted(tile_map.keys())
                if not tiles:
                    ax.set_title(f"Spine {spine}")
                    continue

                ymax = 0
                for tile in tiles:
                    entry = tile_map[tile]
                    ymax = max(ymax, sum(entry.get(bucket, 0) for bucket in buckets))
                if ymax == 0:
                    ymax = 1

                for tile_idx, tile in enumerate(tiles):
                    entry = tile_map[tile]
                    bottom = 0.0
                    for bucket_idx, bucket in enumerate(buckets):
                        height = entry.get(bucket, 0)
                        if height > 0:
                            ax.bar(
                                tile_idx,
                                height,
                                width=0.6,
                                bottom=bottom,
                                color=colors[bucket_idx % len(colors)],
                                edgecolor="black",
                                linewidth=0.3,
                            )
                            bottom += height

                ax.set_xticks(range(len(tiles)))
                ax.set_xticklabels([str(tile) for tile in tiles])
                ax.set_xlabel("Tile ID")
                ax.set_ylabel("Count")
                ax.set_ylim(0, ymax * 1.1)
                ax.grid(axis="y", linestyle="--", alpha=0.25)
                ax.set_title(f"Spine {spine}")

            fig.legend(
                handles=legend_patches,
                labels=[p.get_label() for p in legend_patches],
                loc="upper center",
                ncol=len(legend_patches),
                frameon=False,
                bbox_to_anchor=(0.5, 0.99),
            )
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Wrote {pdf_path}")


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    buckets, data = load_summary(csv_path)
    make_plots(csv_path, buckets, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
