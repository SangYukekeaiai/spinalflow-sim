#!/usr/bin/env python3
"""
Generate flattened timestep duration plots (Histogram, Empirical CDF, and PDF)
from the CSVs produced under a ts_duration directory. Generate these CSVs by
running the simulator with `--ts-duration-csv=on`.

Defaults:
  - Input dir: build_test_l5/stats/repo4/vgg16/ts_duration
  - Outputs:   <input>/hist/layer<ID>[trim].pdf, <input>/cdf/layer<ID>[trim].pdf, <input>/pdf/layer<ID>[trim].pdf

Usage examples:
  - python3 scripts/ts_duration_plots.py
  - python3 scripts/ts_duration_plots.py --input-dir path/to/ts_duration
  - python3 scripts/ts_duration_plots.py --bins 50

Notes:
  - The script flattens all timestep columns (t0, t1, ...) across rows and ignores
    the 'avg' row if present.
  - The PDF plot uses a Gaussian KDE if SciPy is available; otherwise it falls back
    to a simple Gaussian KDE implemented with NumPy.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def _extract_layer_id_from_filename(fname: str) -> str:
    m = re.search(r"layer_(\d+)\.csv$", fname)
    return m.group(1) if m else "unknown"


def _read_flattened_timesteps(csv_path: str) -> List[float]:
    """Read CSV and flatten all timestep columns (t0, t1, ...) into a single list.

    Skips 'avg' rows if present.
    """
    flattened: List[float] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        # Identify timestep columns (any header starting with 't')
        ts_cols = [c for c in reader.fieldnames or [] if c and c.startswith("t")]
        # Fallback: if no t* columns, try all numeric columns except id/label
        if not ts_cols:
            ts_cols = [c for c in (reader.fieldnames or []) if c not in ("output_spine_id", "avg")]

        for row in reader:
            # Skip aggregate row if present
            spine_id = str(row.get("output_spine_id", "")).strip().lower()
            if spine_id == "avg":
                continue
            for c in ts_cols:
                val = str(row.get(c, "")).strip()
                if not val:
                    continue
                try:
                    flattened.append(float(val))
                except ValueError:
                    # Ignore unparsable cells
                    continue
    return flattened


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _compute_ecdf(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n + 1) / n
    return x, y


def _silverman_bandwidth(x: np.ndarray) -> float:
    # Silverman's rule of thumb
    n = x.size
    if n < 2:
        return 1.0
    std = np.std(x, ddof=1) if n > 1 else np.std(x)
    if std <= 0:
        # Fallback to IQR-based or small epsilon
        q75, q25 = np.percentile(x, [75, 25])
        iqr = q75 - q25
        scale = iqr / 1.34 if iqr > 0 else 1.0
        std = scale if scale > 0 else 1.0
    h = 1.06 * std * (n ** (-1 / 5))
    return max(h, 1e-9)


def _kde_gaussian(x: np.ndarray, grid: np.ndarray, bandwidth: float) -> np.ndarray:
    # Simple Gaussian KDE computed in a vectorized manner
    # density(z) = (1/(n*h*sqrt(2pi))) * sum(exp(-0.5*((z - xi)/h)^2))
    n = x.size
    if n == 0:
        return np.zeros_like(grid)
    inv = 1.0 / (bandwidth * math.sqrt(2 * math.pi) * n)
    # Broadcast: grid[:,None] vs x[None,:]
    u = (grid[:, None] - x[None, :]) / bandwidth
    # To guard against potential large arrays, compute in chunks if necessary
    # but for typical sizes this should be fine.
    dens = inv * np.exp(-0.5 * (u ** 2)).sum(axis=1)
    return dens


def _clip_array(arr: np.ndarray,
                xmin: Optional[float], xmax: Optional[float],
                qleft: Optional[float], qright: Optional[float]) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo = -np.inf
    hi = np.inf
    if qleft is not None:
        try:
            ql = float(qleft)
            ql = 0.0 if ql < 0 else (1.0 if ql > 1.0 else ql)
            lo = max(lo, float(np.quantile(arr, ql)))
        except Exception:
            pass
    if qright is not None:
        try:
            qr = float(qright)
            qr = 0.0 if qr < 0 else (1.0 if qr > 1.0 else qr)
            hi = min(hi, float(np.quantile(arr, qr)))
        except Exception:
            pass
    if xmin is not None:
        try:
            lo = max(lo, float(xmin))
        except Exception:
            pass
    if xmax is not None:
        try:
            hi = min(hi, float(xmax))
        except Exception:
            pass
    if not np.isfinite(lo):
        lo = -np.inf
    if not np.isfinite(hi):
        hi = np.inf
    if lo == -np.inf and hi == np.inf:
        return arr
    return arr[(arr >= lo) & (arr <= hi)]


def _clip_suffix(xmin: Optional[float], xmax: Optional[float], qleft: Optional[float], qright: Optional[float]) -> str:
    parts: List[str] = []
    def fmt(v: float) -> str:
        s = f"{v:.6g}"
        return s.replace(" ", "").replace("..", ".").replace("/", "_")
    if xmin is not None:
        parts.append(f"xl{fmt(float(xmin))}")
    if xmax is not None:
        parts.append(f"xr{fmt(float(xmax))}")
    if qleft is not None:
        parts.append(f"ql{fmt(float(qleft))}")
    if qright is not None:
        parts.append(f"qr{fmt(float(qright))}")
    return ("_" + "_".join(parts)) if parts else ""


def _clip_human_label(xmin: Optional[float], xmax: Optional[float], qleft: Optional[float], qright: Optional[float]) -> str:
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


def _fmt_dims(dims: Optional[Tuple[int, int, int]]) -> str:
    if dims is None:
        return ""
    c, h, w = dims
    return f" (Cin={c}, Hin={h}, Win={w})"


def _plot_hist(data: np.ndarray, out_path: str, layer_id: str, bins: int | str = "auto", dims: Optional[Tuple[int, int, int]] = None, title_suffix: str = "") -> None:
    _ensure_dir(os.path.dirname(out_path))
    plt.figure(figsize=(7, 5))
    plt.hist(data, bins=bins, color="#4C78A8", edgecolor="white")
    plt.title(f"Layer {layer_id}: Histogram of Timesteps (Flattened){_fmt_dims(dims)}{title_suffix}")
    plt.xlabel("Timestep duration")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.2, linestyle=":")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_ecdf(data: np.ndarray, out_path: str, layer_id: str, dims: Optional[Tuple[int, int, int]] = None, title_suffix: str = "") -> None:
    _ensure_dir(os.path.dirname(out_path))
    x, y = _compute_ecdf(data)
    plt.figure(figsize=(7, 5))
    plt.step(x, y, where="post", color="#F58518")
    plt.title(f"Layer {layer_id}: Empirical CDF of Timesteps (Flattened){_fmt_dims(dims)}{title_suffix}")
    plt.xlabel("Timestep duration")
    plt.ylabel("ECDF")
    plt.grid(True, alpha=0.2, linestyle=":")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_pdf(data: np.ndarray, out_path: str, layer_id: str, dims: Optional[Tuple[int, int, int]] = None, title_suffix: str = "") -> None:
    _ensure_dir(os.path.dirname(out_path))
    x = np.asarray(data)
    if x.size == 0:
        # Create an empty plot with a note
        plt.figure(figsize=(7, 5))
        plt.title(f"Layer {layer_id}: PDF (no data){_fmt_dims(dims)}{title_suffix}")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        return

    xmin, xmax = np.min(x), np.max(x)
    if xmin == xmax:
        xmin -= 0.5
        xmax += 0.5
    grid = np.linspace(xmin, xmax, 512)

    # Try SciPy if available for KDE, else fallback
    dens = None
    try:
        from scipy.stats import gaussian_kde  # type: ignore
        kde = gaussian_kde(x)
        dens = kde(grid)
    except Exception:
        bw = _silverman_bandwidth(x)
        dens = _kde_gaussian(x, grid, bw)

    plt.figure(figsize=(7, 5))
    plt.plot(grid, dens, color="#54A24B")
    plt.title(f"Layer {layer_id}: PDF of Timesteps (Flattened){_fmt_dims(dims)}{title_suffix}")
    plt.xlabel("Timestep duration")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.2, linestyle=":")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _load_dims_from_json(path: str) -> Dict[str, Tuple[int, int, int]]:
    mapping: Dict[str, Tuple[int, int, int]] = {}
    try:
        with open(path, "r") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):
                    try:
                        c = int(v.get("Cin"))
                        h = int(v.get("Hin"))
                        w = int(v.get("Win"))
                        mapping[str(k)] = (c, h, w)
                    except Exception:
                        continue
    except Exception:
        pass
    return mapping


def _parse_dims_inline(items: List[str]) -> Dict[str, Tuple[int, int, int]]:
    # format: "L:Cin,Hin,Win"
    mapping: Dict[str, Tuple[int, int, int]] = {}
    for it in items:
        try:
            lhs, rhs = it.split(":", 1)
            parts = [p for p in re.split(r"[,xX]", rhs) if p]
            if len(parts) != 3:
                continue
            c, h, w = (int(parts[0]), int(parts[1]), int(parts[2]))
            mapping[str(int(lhs))] = (c, h, w)
        except Exception:
            continue
    return mapping


def _plot_box_across_layers(data_by_layer: Dict[str, np.ndarray],
                            dims_map: Dict[str, Tuple[int, int, int]] | None,
                            out_path: str,
                            trim_label: str = "") -> None:
    _ensure_dir(os.path.dirname(out_path))
    # Sort layers numerically if possible
    def _key(s: str) -> int:
        try:
            return int(s)
        except Exception:
            return 0
    layer_ids = sorted(list(data_by_layer.keys()), key=_key)
    data = [np.asarray(data_by_layer[L], dtype=float) for L in layer_ids]

    # Slightly wider figure to make room for an outside legend
    plt.figure(figsize=(10, 5))
    # Matplotlib >=3.9 renamed 'labels' -> 'tick_labels'. Try new name, fallback for older versions.
    try:
        bp = plt.boxplot(
            data,
            patch_artist=True,
            tick_labels=[f"L{L}" for L in layer_ids],
            showfliers=False,
        )
    except TypeError:
        bp = plt.boxplot(
            data,
            patch_artist=True,
            labels=[f"L{L}" for L in layer_ids],
            showfliers=False,
        )

    colors = list(plt.get_cmap('tab10').colors)
    while len(colors) < len(layer_ids):
        colors.extend(colors)

    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.6)
        patch.set_edgecolor('#444444')
    for whisk in bp['whiskers']:
        whisk.set_color('#666666')
    for cap in bp['caps']:
        cap.set_color('#666666')
    for med in bp['medians']:
        med.set_color('#000000')

    # Build legend entries labeled by configuration
    handles = []
    labels = []
    for i, L in enumerate(layer_ids):
        if dims_map and L in dims_map:
            c, h, w = dims_map[L]
            lab = f"L{L}: Cin={c}, Hin={h}, Win={w}"
        else:
            lab = f"L{L}"
        handles.append(mpatches.Patch(color=colors[i], label=lab, alpha=0.6))
        labels.append(lab)

    # Place legend outside to avoid covering the boxes
    ax = plt.gca()
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True)
    plt.title("Timestep Duration Boxplot Across Layers" + trim_label)
    plt.xlabel("Layer")
    plt.ylabel("Timestep duration")
    plt.grid(True, axis='y', alpha=0.2, linestyle=':')
    # Reserve space for legend and save tightly
    plt.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()


def process_dir(input_dir: str,
                bins: int | str = "auto",
                dims_json: Optional[str] = None,
                dims_inline: Optional[List[str]] = None,
                xmin: Optional[float] = None,
                xmax: Optional[float] = None,
                qleft: Optional[float] = None,
                qright: Optional[float] = None,
                plots: Optional[List[str]] = None) -> None:
    # Discover layer CSVs
    files = [f for f in os.listdir(input_dir) if re.match(r"layer_\d+\.csv$", f)]
    files.sort(key=lambda s: int(re.search(r"\d+", s).group(0)) if re.search(r"\d+", s) else 0)

    if not files:
        print(f"No layer_*.csv files found in {input_dir}", file=sys.stderr)
        return

    hist_dir = os.path.join(input_dir, "hist")
    cdf_dir = os.path.join(input_dir, "cdf")
    pdf_dir = os.path.join(input_dir, "pdf")
    box_dir = os.path.join(input_dir, "box")
    _ensure_dir(hist_dir)
    _ensure_dir(cdf_dir)
    _ensure_dir(pdf_dir)
    _ensure_dir(box_dir)

    # Build dims mapping
    dims_map: Dict[str, Tuple[int, int, int]] = {}
    # 1) explicit JSON flag
    if dims_json:
        dims_map.update(_load_dims_from_json(dims_json))
    # 2) local default JSON in input dir
    default_json = os.path.join(input_dir, "layer_dims.json")
    if not dims_map and os.path.isfile(default_json):
        dims_map.update(_load_dims_from_json(default_json))
    # 3) one level up (model dir)
    model_json = os.path.join(os.path.dirname(input_dir.rstrip(os.sep)), "layer_dims.json")
    if not dims_map and os.path.isfile(model_json):
        dims_map.update(_load_dims_from_json(model_json))
    # 4) inline overrides/appends
    if dims_inline:
        dims_map.update(_parse_dims_inline(dims_inline))

    data_by_layer: Dict[str, np.ndarray] = {}
    suffix = _clip_suffix(xmin, xmax, qleft, qright)
    trim_label = _clip_human_label(xmin, xmax, qleft, qright)
    wanted = set([p.strip().lower() for p in (plots or ["hist", "cdf", "pdf", "box"])])
    for fname in files:
        layer_id = _extract_layer_id_from_filename(fname)
        csv_path = os.path.join(input_dir, fname)
        data = _read_flattened_timesteps(csv_path)
        arr = np.asarray(data, dtype=float)
        # Tail clipping per layer (by absolute bounds and/or quantiles)
        arr = _clip_array(arr, xmin, xmax, qleft, qright)

        if arr.size == 0:
            print(f"[warn] {fname}: no data points found; generating empty plots")

        # Output filenames: layer<ID>.pdf
        hist_out = os.path.join(hist_dir, f"layer{layer_id}{suffix}.pdf")
        cdf_out  = os.path.join(cdf_dir,  f"layer{layer_id}{suffix}.pdf")
        pdf_out  = os.path.join(pdf_dir,  f"layer{layer_id}{suffix}.pdf")

        dims = dims_map.get(layer_id)
        if "hist" in wanted:
            _plot_hist(arr, hist_out, layer_id, bins=bins, dims=dims, title_suffix=trim_label)
        if "cdf" in wanted:
            _plot_ecdf(arr, cdf_out, layer_id, dims=dims, title_suffix=trim_label)
        if "pdf" in wanted:
            _plot_pdf(arr, pdf_out, layer_id, dims=dims, title_suffix=trim_label)

        data_by_layer[layer_id] = arr

        outs = []
        if "hist" in wanted:
            outs.append(os.path.relpath(hist_out))
        if "cdf" in wanted:
            outs.append(os.path.relpath(cdf_out))
        if "pdf" in wanted:
            outs.append(os.path.relpath(pdf_out))
        if outs:
            print("Generated: " + "; ".join(outs))

    # Cross-layer boxplot
    if len(data_by_layer) >= 1 and ("box" in wanted):
        box_out = os.path.join(box_dir, f"all_layers{suffix}.pdf")
        _plot_box_across_layers(data_by_layer, dims_map if dims_map else None, box_out, trim_label)
        print(f"Generated: {os.path.relpath(box_out)}")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Generate histogram/ECDF/PDF plots for ts_duration CSVs.")
    parser.add_argument(
        "--input-dir",
        default=os.path.join("stats", "repo4", "vgg16", "ts_duration"),
        help="Directory containing layer_*.csv files (default: build_test_l5/.../ts_duration)",
    )
    parser.add_argument(
        "--bins",
        type=str,
        default="auto",
        help="Histogram bins (int or 'auto')",
    )
    parser.add_argument(
        "--dims-json",
        default=None,
        help="Path to JSON mapping: {\"<L>\": {\"Cin\":int, \"Hin\":int, \"Win\":int}, ...}. If omitted, script looks for layer_dims.json in input dir or its parent.",
    )
    parser.add_argument(
        "--dims",
        action="append",
        default=[],
        help="Inline dims mapping as 'L:Cin,Hin,Win' (e.g., --dims 5:64,32,32). Can be repeated.",
    )
    parser.add_argument("--xmin", type=float, default=None, help="Clip values below this (inclusive)")
    parser.add_argument("--xmax", type=float, default=None, help="Clip values above this (inclusive)")
    parser.add_argument("--qleft", type=float, default=None, help="Clip values below this quantile [0..1]")
    parser.add_argument("--qright", type=float, default=None, help="Clip values above this quantile [0..1]")
    parser.add_argument(
        "--plots",
        type=str,
        default="hist,cdf,pdf,box",
        help="Comma-separated list of plots to generate: hist,cdf,pdf,box",
    )

    args = parser.parse_args(argv)
    bins: int | str
    if args.bins.isdigit():
        bins = int(args.bins)
    else:
        bins = args.bins

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    plots = [p for p in args.plots.split(',') if p]
    process_dir(input_dir,
                bins=bins,
                dims_json=args.dims_json,
                dims_inline=args.dims,
                xmin=args.xmin,
                xmax=args.xmax,
                qleft=args.qleft,
                qright=args.qright,
                plots=plots)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
