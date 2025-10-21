#!/usr/bin/env python3
"""
Generate per-layer plots (Histogram, Empirical CDF, PDF via KDE) and a cross-layer
boxplot from reuse-distance distribution CSVs under a reuse_distance_distribution directory.

Defaults:
  - Input dir: build_test_l5/stats/repo4/vgg16/reuse_distance_distribution
  - Outputs:   <input>/hist/layer<ID>[trim].pdf, <input>/cdf/layer<ID>[trim].pdf,
                <input>/pdf/layer<ID>[trim].pdf, <input>/box/all_layers[trim].pdf

CSV format (per layer):
  reuse_distance,count,share
  <int>,<int>,<float>

Notes:
  - Histogram is rendered directly from (reuse_distance,count) as a bar plot.
  - ECDF and PDF are derived using weights (counts), so no full sample expansion is required.
  - The boxplot across layers uses weighted quantiles to compute the five-number summary.
  - Plot titles optionally include layer dims (Cin,Hin,Win) if `layer_dims.json` is found.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def _extract_layer_id_from_filename(fname: str) -> str:
    m = re.search(r"layer(_|)(\d+)\.csv$", fname)
    if not m:
        m = re.search(r"layer_(\d+)\.csv$", fname)
    return m.group(2) if m and m.lastindex and m.lastindex >= 2 else (m.group(1) if m else "unknown")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _fmt_dims(dims: Optional[Tuple[int, int, int]]) -> str:
    if dims is None:
        return ""
    c, h, w = dims
    return f" (Cin={c}, Hin={h}, Win={w})"


def _read_reuse_hist(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[float] = []
    ws: List[float] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = float(row.get("reuse_distance", ""))
                w = float(row.get("count", ""))
            except Exception:
                continue
            if not np.isfinite(x) or not np.isfinite(w):
                continue
            if w <= 0:
                continue
            xs.append(x)
            ws.append(w)
    if not xs:
        return np.array([], dtype=float), np.array([], dtype=float)
    arr_x = np.asarray(xs, dtype=float)
    arr_w = np.asarray(ws, dtype=float)
    # Sort by x ascending
    idx = np.argsort(arr_x)
    return arr_x[idx], arr_w[idx]


def _plot_hist_bar(x: np.ndarray, w: np.ndarray, out_path: str, title: str) -> None:
    _ensure_dir(os.path.dirname(out_path))
    plt.figure(figsize=(8, 4.5))
    plt.bar(x, w, width=1.0, color="#4C78A8", edgecolor="none")
    plt.title(title)
    plt.xlabel("Reuse distance")
    plt.ylabel("Count")
    plt.grid(True, axis="y", alpha=0.2, linestyle=":")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_ecdf_weighted(x: np.ndarray, w: np.ndarray, out_path: str, title: str) -> None:
    _ensure_dir(os.path.dirname(out_path))
    if x.size == 0:
        plt.figure(figsize=(8, 4.5))
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        return
    wsum = np.sum(w)
    cdf = np.cumsum(w) / wsum
    plt.figure(figsize=(8, 4.5))
    plt.step(x, cdf, where="post", color="#F58518")
    plt.title(title)
    plt.xlabel("Reuse distance")
    plt.ylabel("ECDF")
    plt.grid(True, alpha=0.2, linestyle=":")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _silverman_bandwidth(x: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    n = x.size
    if n < 2:
        return 1.0
    if w is None:
        std = np.std(x, ddof=1) if n > 1 else np.std(x)
    else:
        # Weighted std
        wsum = np.sum(w)
        mean = np.sum(w * x) / wsum
        var = np.sum(w * (x - mean) ** 2) / wsum
        std = math.sqrt(max(var, 0.0))
    if std <= 0:
        q75, q25 = np.percentile(x, [75, 25])
        iqr = q75 - q25
        std = (iqr / 1.34) if iqr > 0 else 1.0
    h = 1.06 * std * (n ** (-1 / 5))
    return max(h, 1e-9)


def _kde_gaussian_weighted(x: np.ndarray, w: np.ndarray, grid: np.ndarray, bandwidth: float) -> np.ndarray:
    if x.size == 0:
        return np.zeros_like(grid)
    wsum = np.sum(w)
    inv = 1.0 / (bandwidth * math.sqrt(2 * math.pi))
    u = (grid[:, None] - x[None, :]) / bandwidth
    kern = np.exp(-0.5 * (u ** 2))
    dens = inv * (kern * (w[None, :] / wsum)).sum(axis=1)
    return dens


def _plot_pdf_weighted(x: np.ndarray, w: np.ndarray, out_path: str, title: str) -> None:
    _ensure_dir(os.path.dirname(out_path))
    if x.size == 0:
        plt.figure(figsize=(8, 4.5))
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        return
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmin == xmax:
        xmin -= 0.5
        xmax += 0.5
    grid = np.linspace(xmin, xmax, 512)
    try:
        from scipy.stats import gaussian_kde  # type: ignore
        kde = gaussian_kde(x, weights=w)
        dens = kde(grid)
    except Exception:
        bw = _silverman_bandwidth(x, w)
        dens = _kde_gaussian_weighted(x, w, grid, bw)
    plt.figure(figsize=(8, 4.5))
    plt.plot(grid, dens, color="#54A24B")
    plt.title(title)
    plt.xlabel("Reuse distance")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.2, linestyle=":")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    if x.size == 0:
        return float('nan')
    if q <= 0:
        return float(x[0])
    if q >= 1:
        return float(x[-1])
    cw = np.cumsum(w)
    target = q * cw[-1]
    idx = np.searchsorted(cw, target, side='left')
    idx = min(max(idx, 0), x.size - 1)
    return float(x[idx])


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


def _clip_weighted(x: np.ndarray, w: np.ndarray,
                   xmin: Optional[float], xmax: Optional[float],
                   qleft: Optional[float], qright: Optional[float]) -> Tuple[np.ndarray, np.ndarray]:
    if x.size == 0:
        return x, w
    lo = -np.inf
    hi = np.inf
    if qleft is not None:
        try:
            ql = float(qleft)
            ql = 0.0 if ql < 0 else (1.0 if ql > 1.0 else ql)
            lo = max(lo, _weighted_quantile(x, w, ql))
        except Exception:
            pass
    if qright is not None:
        try:
            qr = float(qright)
            qr = 0.0 if qr < 0 else (1.0 if qr > 1.0 else qr)
            hi = min(hi, _weighted_quantile(x, w, qr))
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
        return x, w
    m = (x >= lo) & (x <= hi)
    return x[m], w[m]


def _plot_box_across_layers(stats_by_layer: Dict[str, Tuple[np.ndarray, np.ndarray]],
                            dims_map: Dict[str, Tuple[int, int, int]] | None,
                            out_path: str,
                            trim_label: str = "") -> None:
    _ensure_dir(os.path.dirname(out_path))
    def _key(s: str) -> int:
        try:
            return int(s)
        except Exception:
            return 0
    layer_ids = sorted(list(stats_by_layer.keys()), key=_key)

    # Prepare bxp stats (five-number summary using weighted quantiles)
    bxp_stats = []
    colors = list(plt.get_cmap('tab10').colors)
    while len(colors) < len(layer_ids):
        colors.extend(colors)

    legend_handles = []
    for i, L in enumerate(layer_ids):
        x, w = stats_by_layer[L]
        q1 = _weighted_quantile(x, w, 0.25)
        med = _weighted_quantile(x, w, 0.50)
        q3 = _weighted_quantile(x, w, 0.75)
        whislo = float(x[0]) if x.size else float('nan')
        whishi = float(x[-1]) if x.size else float('nan')
        bxp_stats.append({
            'label': f"L{L}",
            'whislo': whislo, 'q1': q1, 'med': med, 'q3': q3, 'whishi': whishi,
        })
        if dims_map and L in dims_map:
            c, h, w3 = dims_map[L]
            lab = f"L{L}: Cin={c}, Hin={h}, Win={w3}"
        else:
            lab = f"L{L}"
        legend_handles.append(mpatches.Patch(color=colors[i], label=lab, alpha=0.6))

    # Slightly wider figure to make room for an outside legend
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    bxp = ax.bxp(bxp_stats, patch_artist=True, showfliers=False)
    for i, box in enumerate(bxp['boxes']):
        box.set_facecolor(colors[i])
        box.set_alpha(0.6)
        box.set_edgecolor('#444444')
    for med in bxp['medians']:
        med.set_color('#000000')
    for whisk in bxp['whiskers']:
        whisk.set_color('#666666')
    for cap in bxp['caps']:
        cap.set_color('#666666')

    # Place legend outside to avoid covering the boxes
    # Anchor on the right side, vertically centered.
    ax.legend(
        handles=legend_handles,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
    )
    plt.title("Reuse Distance Boxplot Across Layers" + trim_label)
    plt.xlabel("Layer")
    plt.ylabel("Reuse distance")
    plt.grid(True, axis='y', alpha=0.2, linestyle=':')
    # Reserve space for the outside legend and save tightly.
    plt.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    plt.savefig(out_path, bbox_inches='tight')
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
                        mapping[str(k)] = (int(v.get('Cin')), int(v.get('Hin')), int(v.get('Win')))
                    except Exception:
                        continue
    except Exception:
        pass
    return mapping


def _parse_dims_inline(items: List[str]) -> Dict[str, Tuple[int, int, int]]:
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


def process_dir(input_dir: str,
                dims_json: Optional[str] = None,
                dims_inline: Optional[List[str]] = None,
                xmin: Optional[float] = None,
                xmax: Optional[float] = None,
                qleft: Optional[float] = None,
                qright: Optional[float] = None,
                plots: Optional[List[str]] = None) -> None:
    files = [f for f in os.listdir(input_dir) if re.match(r"layer_?\d+\.csv$", f)]
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

    # Dims discovery
    dims_map: Dict[str, Tuple[int, int, int]] = {}
    if dims_json:
        dims_map.update(_load_dims_from_json(dims_json))
    local_json = os.path.join(input_dir, "layer_dims.json")
    if not dims_map and os.path.isfile(local_json):
        dims_map.update(_load_dims_from_json(local_json))
    # Parent dir
    parent_json = os.path.join(os.path.dirname(input_dir.rstrip(os.sep)), "layer_dims.json")
    if not dims_map and os.path.isfile(parent_json):
        dims_map.update(_load_dims_from_json(parent_json))
    # Try ts_duration sibling
    model_dir = os.path.dirname(input_dir.rstrip(os.sep))
    ts_json = os.path.join(model_dir, "ts_duration", "layer_dims.json")
    if os.path.isfile(ts_json):
        dims_map.update(_load_dims_from_json(ts_json))
    if dims_inline:
        dims_map.update(_parse_dims_inline(dims_inline))

    stats_by_layer: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    suffix = _clip_suffix(xmin, xmax, qleft, qright)
    trim_label = _clip_human_label(xmin, xmax, qleft, qright)
    wanted = set([p.strip().lower() for p in (plots or ["hist", "cdf", "pdf", "box"])])

    for fname in files:
        layer_id = re.search(r"\d+", fname).group(0)
        csv_path = os.path.join(input_dir, fname)
        x, w = _read_reuse_hist(csv_path)
        # Tail clipping by absolute bounds and/or weighted quantiles
        x, w = _clip_weighted(x, w, xmin, xmax, qleft, qright)
        dims = dims_map.get(layer_id)

        # Outputs
        hist_out = os.path.join(hist_dir, f"layer{layer_id}{suffix}.pdf")
        cdf_out  = os.path.join(cdf_dir,  f"layer{layer_id}{suffix}.pdf")
        pdf_out  = os.path.join(pdf_dir,  f"layer{layer_id}{suffix}.pdf")

        title_base = f"Layer {layer_id}{_fmt_dims(dims)}" + trim_label
        if "hist" in wanted:
            _plot_hist_bar(x, w, hist_out, title_base + ": Reuse Distance Histogram")
        if "cdf" in wanted:
            _plot_ecdf_weighted(x, w, cdf_out, title_base + ": Empirical CDF")
        if "pdf" in wanted:
            _plot_pdf_weighted(x, w, pdf_out, title_base + ": PDF (KDE)")

        stats_by_layer[layer_id] = (x, w)
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
    if stats_by_layer and ("box" in wanted):
        box_out = os.path.join(box_dir, f"all_layers{suffix}.pdf")
        _plot_box_across_layers(stats_by_layer, dims_map if dims_map else None, box_out, trim_label)
        print(f"Generated: {os.path.relpath(box_out)}")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Plot reuse-distance distributions per layer and across layers.")
    parser.add_argument(
        "--input-dir",
        default=os.path.join("stats", "repo4", "vgg16", "reuse_distance_distribution"),
        help="Directory containing layer_*.csv files (default: build_test_l5/.../reuse_distance_distribution)",
    )
    parser.add_argument(
        "--dims-json",
        default=None,
        help="Path to JSON mapping: {\"<L>\": {\"Cin\":int, \"Hin\":int, \"Win\":int}, ...}. If omitted, script looks for layer_dims.json nearby or under ts_duration/.",
    )
    parser.add_argument(
        "--dims",
        action="append",
        default=[],
        help="Inline dims mapping as 'L:Cin,Hin,Win' (e.g., --dims 5:64,32,32). Can be repeated.",
    )
    parser.add_argument("--xmin", type=float, default=None, help="Clip reuse distances below this (inclusive)")
    parser.add_argument("--xmax", type=float, default=None, help="Clip reuse distances above this (inclusive)")
    parser.add_argument("--qleft", type=float, default=None, help="Clip below this weighted quantile [0..1]")
    parser.add_argument("--qright", type=float, default=None, help="Clip above this weighted quantile [0..1]")
    parser.add_argument(
        "--plots",
        type=str,
        default="hist,cdf,pdf,box",
        help="Comma-separated list of plots to generate: hist,cdf,pdf,box",
    )
    args = parser.parse_args(argv)

    input_dir = args.input_dir
    if not os.path.isdir(input_dir):
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    plots = [p for p in args.plots.split(',') if p]
    process_dir(input_dir,
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
