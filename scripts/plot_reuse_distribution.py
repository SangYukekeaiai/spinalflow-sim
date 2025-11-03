#!/usr/bin/env python3
"""Create a cross-layer reuse-distance boxplot that mirrors reuse_distance_plots.py."""

import argparse
import csv
import json
import pathlib
import re
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


LAYER_RE = re.compile(r"layer_?(\d+)\.csv$")


def _ensure_dir(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _parse_layer_id(path: pathlib.Path) -> int:
    match = LAYER_RE.match(path.name)
    if match:
        return int(match.group(1))
    # If the filename is something else, park it at the end.
    digits = re.findall(r"\d+", path.stem)
    return int(digits[0]) if digits else 10**9


def _read_hist(csv_path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[float] = []
    ws: List[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = float(row.get("reuse_distance", ""))
                w = float(row.get("count", ""))
            except Exception:
                continue
            if not np.isfinite(x) or not np.isfinite(w) or w <= 0:
                continue
            xs.append(x)
            ws.append(w)
    if not xs:
        return np.array([]), np.array([])
    arr_x = np.asarray(xs, dtype=float)
    arr_w = np.asarray(ws, dtype=float)
    order = np.argsort(arr_x)
    return arr_x[order], arr_w[order]


def _weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    if x.size == 0:
        return float("nan")
    if q <= 0:
        return float(x[0])
    if q >= 1:
        return float(x[-1])
    cumulative = np.cumsum(w)
    target = q * cumulative[-1]
    idx = np.searchsorted(cumulative, target, side="left")
    idx = min(max(idx, 0), x.size - 1)
    return float(x[idx])


def _clip_weighted(x: np.ndarray, w: np.ndarray,
                   xmin: Optional[float], xmax: Optional[float],
                   qleft: Optional[float], qright: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
    if x.size == 0:
        return x, w
    lo = -np.inf
    hi = np.inf
    if qleft is not None:
        ql = min(max(qleft, 0.0), 1.0)
        lo = max(lo, _weighted_quantile(x, w, ql))
    if qright is not None:
        qr = min(max(qright, 0.0), 1.0)
        hi = min(hi, _weighted_quantile(x, w, qr))
    if xmin is not None:
        lo = max(lo, xmin)
    if xmax is not None:
        hi = min(hi, xmax)
    if not np.isfinite(lo) and not np.isfinite(hi):
        return x, w
    mask = (x >= lo) & (x <= hi)
    return x[mask], w[mask]


def _plot_box(stats_by_layer: Dict[int, Tuple[np.ndarray, np.ndarray]],
              dims_map: Optional[Dict[int, Tuple[int, int, int]]],
              output: pathlib.Path,
              clip_label: str) -> None:
    _ensure_dir(output.parent)

    layer_ids = sorted(stats_by_layer.keys())
    if not layer_ids:
        raise SystemExit("No reuse distance data available for plotting.")

    colors = list(plt.get_cmap("tab10").colors)
    while len(colors) < len(layer_ids):
        colors.extend(colors)
    plot_colors: List[str] = []

    bxp_stats = []
    legend_handles = []
    for idx, lid in enumerate(layer_ids):
        x, w = stats_by_layer[lid]
        if x.size == 0:
            continue
        q1 = _weighted_quantile(x, w, 0.25)
        med = _weighted_quantile(x, w, 0.50)
        q3 = _weighted_quantile(x, w, 0.75)
        color = colors[idx % len(colors)]
        bxp_stats.append({
            "label": f"L{lid}",
            "whislo": float(x[0]),
            "q1": q1,
            "med": med,
            "q3": q3,
            "whishi": float(x[-1]),
        })
        plot_colors.append(color)
        if dims_map and lid in dims_map:
            cin, hin, win = dims_map[lid]
            label = f"Layer {lid}: Cin={cin}, Hin={hin}, Win={win}"
        else:
            label = f"Layer {lid}"
        legend_handles.append(mpatches.Patch(color=color, label=label, alpha=0.6))

    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    boxplot = ax.bxp(bxp_stats, patch_artist=True, showfliers=False)

    for color, box in zip(plot_colors, boxplot["boxes"]):
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
    ax.set_title(f"Reuse Distance Boxplot Across Layers{clip_label}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Reuse distance")
    ax.grid(True, axis="y", alpha=0.2, linestyle=":")
    plt.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    plt.savefig(output, bbox_inches="tight")
    plt.close()


def _clip_label(xmin: Optional[float], xmax: Optional[float],
                qleft: Optional[float], qright: Optional[float]) -> str:
    parts: List[str] = []
    if xmin is not None:
        parts.append(f"xl={xmin}")
    if xmax is not None:
        parts.append(f"xr={xmax}")
    if qleft is not None:
        parts.append(f"ql={qleft}")
    if qright is not None:
        parts.append(f"qr={qright}")
    return (" [trim: " + ", ".join(parts) + "]") if parts else ""


def _load_dims(input_dir: pathlib.Path) -> Dict[int, Tuple[int, int, int]]:
    candidates = [
        input_dir / "layer_dims.json",
        input_dir.parent / "layer_dims.json",
        input_dir.parent / "ts_duration" / "layer_dims.json",
    ]
    dims_map: Dict[int, Tuple[int, int, int]] = {}
    for path in candidates:
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                for key, value in data.items():
                    if not isinstance(value, dict):
                        continue
                    try:
                        lid = int(key)
                        dims_map[lid] = (
                            int(value.get("Cin")),
                            int(value.get("Hin")),
                            int(value.get("Win")),
                        )
                    except Exception:
                        continue
        except Exception:
            continue
        if dims_map:
            break
    return dims_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot reuse distance boxplot across layers")
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing layer*.csv reuse histograms")
    parser.add_argument("--output",
                        help="Output PDF path (default: <input-dir>/box/reuse_distance_boxplot.pdf)")
    parser.add_argument("--xmin", type=float, default=None,
                        help="Clip reuse distances below this value before plotting")
    parser.add_argument("--xmax", type=float, default=None,
                        help="Clip reuse distances above this value before plotting")
    parser.add_argument("--qleft", type=float, default=None,
                        help="Clip below this weighted quantile (0-1 range)")
    parser.add_argument("--qright", type=float, default=None,
                        help="Clip above this weighted quantile (0-1 range)")
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    csv_files = sorted(input_dir.glob("layer*.csv"), key=_parse_layer_id)
    if not csv_files:
        raise SystemExit("No layer*.csv files found in input directory.")

    stats_by_layer: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for csv_path in csv_files:
        x, w = _read_hist(csv_path)
        x, w = _clip_weighted(x, w, args.xmin, args.xmax, args.qleft, args.qright)
        if x.size == 0:
            continue
        stats_by_layer[_parse_layer_id(csv_path)] = (x, w)

    clip_label = _clip_label(args.xmin, args.xmax, args.qleft, args.qright)

    output_path = pathlib.Path(args.output) if args.output else (
        input_dir / "box" / "reuse_distance_boxplot.pdf"
    )

    dims_map = _load_dims(input_dir)
    _plot_box(stats_by_layer, dims_map, output_path, clip_label)
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    main()
