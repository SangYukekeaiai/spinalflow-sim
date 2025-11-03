#!/usr/bin/env python3
"""
Plot eviction quality distribution per output spine.

Input CSV format (columns produced by --evict-quality):
  output spine id,tile id,time steps,eviction total,
  bad eviction 1 times,bad eviction 2 times,...

For each output spine id this script creates a chart where the x-axis lists tile
ids. For every tile, timesteps are drawn as adjacent stacked bars whose
segments show good evictions (total - bad total) and bad evictions by the delay
bucket (first reuse after 1 timestep, 2 timesteps, etc.). The resulting PNGs
are written next to the CSV.
"""

from __future__ import annotations

import argparse
import csv
import sys
import io
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot eviction quality distribution per output spine.",
    )
    parser.add_argument(
        "csv_path",
        help="Path to evict_quality_distribution.csv",
    )
    return parser.parse_args()


def read_distribution(csv_path: Path):
    """Load CSV into nested dictionaries.

    Returns:
        categories: list of bad eviction columns (strings)
        data: dict[spine][tile][timestep] -> dict of counts
    """
    data: Dict[str, Dict[int, Dict[int, Dict[str, int]]]] = defaultdict(
        lambda: defaultdict(dict)
    )

    raw_text = csv_path.read_text(encoding="utf-8")
    normalized_text = re.sub(
        r"\(\s*(\d+)\s*,\s*(\d+)\s*\)",
        lambda m: f"{m.group(1)}_{m.group(2)}",
        raw_text,
    )

    with io.StringIO(normalized_text) as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV missing header.")

        bad_columns = [
            name
            for name in reader.fieldnames
            if name.lower().startswith("bad eviction")
            and name.lower() != "bad eviction total"
        ]
        bad_columns.sort(key=lambda s: int("".join(filter(str.isdigit, s)) or 0))

        for row in reader:
            try:
                spine_token = row["output spine id"].strip()
                tile = int(row["tile id"])
                timestep = int(row["time steps"])
                ev_total = int(row["eviction total"])
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Malformed row: {row}") from exc

            entry: Dict[str, int] = {}
            bad_sum = 0
            for col in bad_columns:
                entry[col] = int(row.get(col, "0") or 0)
                bad_sum += entry[col]
            entry["good"] = max(ev_total - bad_sum, 0)

            if "_" in spine_token:
                parts = spine_token.split("_")
                if len(parts) == 2 and all(p.lstrip("-").isdigit() for p in parts):
                    label = f"({int(parts[0])}, {int(parts[1])})"
                else:
                    label = spine_token
            else:
                label = spine_token

            data[label][tile][timestep] = entry

    return bad_columns, data


def make_plots(csv_path: Path, bad_columns: List[str], data) -> None:
    """Generate a multi-plot PDF summarizing eviction quality."""
    if not data:
        print("No data to plot.", file=sys.stderr)
        return

    output_dir = csv_path.parent
    colors = ["#4C78A8", "#F58518", "#E4572E", "#72B7B2", "#B279A2", "#FF9DA7"]
    categories = ["good"] + bad_columns
    category_labels = ["Good"] + [col.replace("_", " ").title() for col in bad_columns]
    legend_patches = [
        Patch(facecolor=colors[idx % len(colors)], edgecolor="black", label=category_labels[idx])
        for idx in range(len(categories))
    ]

    def spine_sort_key(label: str):
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

    spines = sorted(data.keys(), key=spine_sort_key)
    if not spines:
        return

    per_page = 8
    cols = 3
    rows = 3

    pdf_path = output_dir / "small_multiple_eviction_quality_distribution.pdf"
    with PdfPages(pdf_path) as pdf:
        for page_start in range(0, len(spines), per_page):
            page_spines = spines[page_start : page_start + per_page]
            nrows = min(rows, math.ceil(len(page_spines) / cols))
            fig, axes = plt.subplots(
                nrows,
                cols,
                figsize=(cols * 3.5, nrows * 3.0),
                squeeze=False,
                sharey=False,
            )

            for ax in axes.flatten():
                ax.axis("off")

            for idx, spine in enumerate(page_spines):
                r = idx // cols
                c = idx % cols
                ax = axes[r][c]
                ax.axis("on")
                tile_map = data[spine]
                tiles = sorted(tile_map.keys())
                if not tiles:
                    ax.set_title(f"Spine {spine}")
                    continue

                y_max = 0
                for tile_id in tiles:
                    for values in tile_map[tile_id].values():
                        y_max = max(y_max, sum(values.get(cat, 0) for cat in categories))
                if y_max == 0:
                    y_max = 1

                for tile_idx, tile_id in enumerate(tiles):
                    timestep_map = tile_map[tile_id]
                    timesteps = sorted(timestep_map.keys())
                    if not timesteps:
                        continue

                    group_width = 0.8
                    bar_width = group_width / max(1, len(timesteps))

                    for ts_idx, ts in enumerate(timesteps):
                        x = tile_idx + (ts_idx - (len(timesteps) - 1) / 2.0) * bar_width
                        values = timestep_map[ts]
                        bottom = 0.0
                        for cat_idx, cat in enumerate(categories):
                            height = values.get(cat, 0)
                            if height > 0:
                                ax.bar(
                                    x,
                                    height,
                                    width=bar_width * 0.85,
                                    bottom=bottom,
                                    color=colors[cat_idx % len(colors)],
                                    edgecolor="black",
                                    linewidth=0.3,
                                )
                                bottom += height

                        ax.text(
                            x,
                            -0.06,
                            f"ts{ts}",
                            ha="center",
                            va="top",
                            fontsize=7,
                            rotation=90,
                            transform=ax.get_xaxis_transform(),
                            clip_on=False,
                        )

                ax.set_xticks(range(len(tiles)))
                ax.set_xticklabels([str(tile) for tile in tiles])
                ax.set_xlabel("Tile ID")
                ax.set_ylabel("Evictions")
                ax.set_ylim(0, y_max * 1.15)
                ax.grid(axis="y", linestyle="--", alpha=0.25)
                ax.set_title(f"Spine {spine}")

            fig.legend(
                handles=legend_patches,
                labels=[patch.get_label() for patch in legend_patches],
                loc="upper center",
                ncol=len(legend_patches),
                frameon=False,
                bbox_to_anchor=(0.5, 0.99),
            )
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Wrote {pdf_path}")


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1

    bad_columns, data = read_distribution(csv_path)
    make_plots(csv_path, bad_columns, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
