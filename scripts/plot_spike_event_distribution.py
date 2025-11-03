#!/usr/bin/env python3
"""Render per-layer spike-count distributions (per time / tile / spine)."""

import argparse
import csv
import json
import pathlib
import re
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


LAYER_RE = re.compile(r"layer_(\d+)\.csv$")


def _ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_dims(input_dir: pathlib.Path) -> Dict[int, tuple[int, int, int]]:
    candidates = [
        input_dir / "layer_dims.json",
        input_dir.parent / "layer_dims.json",
        input_dir.parent / "ts_duration" / "layer_dims.json",
    ]
    dims: Dict[int, tuple[int, int, int]] = {}
    for path in candidates:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        for key, value in data.items():
            if not isinstance(value, dict):
                continue
            try:
                lid = int(key)
                dims[lid] = (
                    int(value.get("Cin")),
                    int(value.get("Hin")),
                    int(value.get("Win")),
                )
            except Exception:
                continue
        if dims:
            break
    return dims


def _parse_layer_id(path: pathlib.Path) -> int:
    match = LAYER_RE.match(path.name)
    if match:
        return int(match.group(1))
    digits = re.findall(r"\d+", path.stem)
    return int(digits[0]) if digits else 10**9


def _load_values(csv_path: pathlib.Path) -> np.ndarray:
    values: List[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or len(header) <= 2:
            return np.array(values)
        for row in reader:
            if len(row) < len(header):
                continue
            for val in row[2:]:
                try:
                    values.append(float(val))
                except Exception:
                    continue
    return np.asarray(values, dtype=float)


def _clip(values: np.ndarray,
          ymin: Optional[float],
          ymax: Optional[float],
          qlow: Optional[float],
          qhigh: Optional[float]) -> np.ndarray:
    if values.size == 0:
        return values
    lo = -np.inf
    hi = np.inf
    if qlow is not None:
        ql = min(max(qlow, 0.0), 1.0)
        lo = max(lo, float(np.quantile(values, ql)))
    if qhigh is not None:
        qh = min(max(qhigh, 0.0), 1.0)
        hi = min(hi, float(np.quantile(values, qh)))
    if ymin is not None:
        lo = max(lo, ymin)
    if ymax is not None:
        hi = min(hi, ymax)
    if not np.isfinite(lo) and not np.isfinite(hi):
        return values
    return values[(values >= lo) & (values <= hi)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-layer spike-count distributions as a boxplot"
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing spiking_event_stats layer_*.csv files",
    )
    parser.add_argument(
        "--output",
        help="Output PDF path (default: <input-dir>/box/spiking_event_boxplot.pdf)",
    )
    parser.add_argument("--ymin", type=float, help="Clip values below this threshold")
    parser.add_argument("--ymax", type=float, help="Clip values above this threshold")
    parser.add_argument("--qleft", type=float, help="Clip below this quantile [0-1]")
    parser.add_argument("--qright", type=float, help="Clip above this quantile [0-1]")
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    files = sorted(input_dir.glob("layer_*.csv"), key=_parse_layer_id)
    if not files:
        raise SystemExit("No layer_*.csv files found.")

    dims_map = _load_dims(input_dir)

    layer_ids: List[int] = []
    layer_values: List[np.ndarray] = []

    for csv_path in files:
        lid = _parse_layer_id(csv_path)
        vals = _load_values(csv_path)
        vals = _clip(vals, args.ymin, args.ymax, args.qleft, args.qright)
        if vals.size == 0:
            continue
        layer_ids.append(lid)
        layer_values.append(vals)

    if not layer_values:
        raise SystemExit("No valid spike statistics found after clipping.")

    colors = list(plt.get_cmap("tab10").colors)
    while len(colors) < len(layer_values):
        colors.extend(colors)

    bxp_stats = []
    legend_handles = []
    for idx, (lid, vals) in enumerate(zip(layer_ids, layer_values)):
        sorted_vals = np.sort(vals)
        q1 = np.quantile(sorted_vals, 0.25)
        med = np.quantile(sorted_vals, 0.50)
        q3 = np.quantile(sorted_vals, 0.75)
        color = colors[idx % len(colors)]
        bxp_stats.append({
            "label": f"L{lid}",
            "whislo": float(sorted_vals[0]),
            "q1": float(q1),
            "med": float(med),
            "q3": float(q3),
            "whishi": float(sorted_vals[-1]),
        })
        if lid in dims_map:
            cin, hin, win = dims_map[lid]
            label = f"Layer {lid}: Cin={cin}, Hin={hin}, Win={win}"
        else:
            label = f"Layer {lid}"
        legend_handles.append(mpatches.Patch(color=color, label=label, alpha=0.6))

    fig, ax = plt.subplots(figsize=(10, 5))
    boxplot = ax.bxp(bxp_stats, patch_artist=True, showfliers=False)
    for color, box in zip(colors[:len(boxplot["boxes"])], boxplot["boxes"]):
        box.set_facecolor(color)
        box.set_alpha(0.6)
        box.set_edgecolor("#444444")
    for median in boxplot["medians"]:
        median.set_color("#000000")
    for whisker in boxplot["whiskers"]:
        whisker.set_color("#666666")
    for cap in boxplot["caps"]:
        cap.set_color("#666666")

    ax.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    ax.set_title("Spike count distribution per layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Spike count per time/tile/spine")
    ax.grid(True, axis="y", alpha=0.2, linestyle=":")
    plt.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])

    output_path = pathlib.Path(args.output) if args.output else (
        input_dir / "box" / "spiking_event_boxplot.pdf"
    )
    _ensure_dir(output_path.parent)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    main()

