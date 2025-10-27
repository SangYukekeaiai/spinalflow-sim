#!/usr/bin/env python3
"""
Visualize eviction quality (good vs. bad evictions) over timesteps and tiles.

Inputs
  - A CSV with header like:
      output_pos(hout, wout), tile_id, timestep, evict total, evict bad, evict good, bad rate, good rate

Behavior
  - Facet by output position (hout,wout): for each timestep draw a cluster of
    one bar per tile (supports 1, 2, or 4 tiles). Each bar is stacked: good
    (lower) + bad (upper).
  - Generate both counts and percent views. Percent shows composition (rates),
    counts shows magnitude.
  - Provide small multiples and overall (across positions+tiles) per‑timestep plots.

Outputs (for CSV path
      stats/repo4/vgg16/layer5/cache_traces/lru/eviction_quality_144KB_4ways_0prefetches.csv)
  - Written under
      stats/repo4/vgg16/layer5/cache_traces/lru/eviction_quality_144KB_4ways_0prefetches/
    with filenames like:
      <hout>_<wout>_evq_counts.pdf, <hout>_<wout>_evq_rates.pdf,
      small_multiples_evq_counts.pdf, small_multiples_evq_rates.pdf,
      overall_per_timestep_evq_counts.pdf, overall_per_timestep_evq_rates.pdf

Usage examples
  - python3 scripts/eviction_quality_plots.py
  - python3 scripts/eviction_quality_plots.py --csv /.../eviction_quality_144KB_4ways_0prefetches.csv
  - python3 scripts/eviction_quality_plots.py --per-page 8 --plots perpos_rates,overall_rates
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter


# Colors aligned with other plotting scripts in this repo
# Provide distinct tile colors for up to 4 tiles; avoids green/red used for components.
TILE_COLORS = [
    "#4C78A8",  # blue (Tile 0)
    "#F58518",  # orange (Tile 1)
    "#B279A2",  # purple (Tile 2)
    "#9D755D",  # brown (Tile 3)
]
COLOR_GOOD = "#9ACD32"    # green-ish
COLOR_BAD = "#CD5C5C"     # red-ish
HATCH_GOOD = "////"
HATCH_BAD = ".."


@dataclass
class Row:
    hout: int
    wout: int
    tile: int
    timestep: int
    ev_total: int
    ev_bad: int
    ev_good: int
    bad_rate: float
    good_rate: float

    @property
    def total(self) -> int:
        return int(self.ev_total)

    @property
    def good_rate_calc(self) -> float:
        t = self.total
        return (self.ev_good / t) if t > 0 else 0.0

    @property
    def bad_rate_calc(self) -> float:
        t = self.total
        return (self.ev_bad / t) if t > 0 else 0.0


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _default_paths() -> Tuple[str, str]:
    csv_path = os.path.join(
        "stats",
        "repo4",
        "vgg16",
        "layer5",
        "cache_traces",
        "lru",
        "eviction_quality_144KB_4ways_0prefetches.csv",
    )
    outdir = os.path.splitext(csv_path)[0]
    return csv_path, outdir


def read_evq_csv(csv_path: str) -> List[Row]:
    """Robust reader for eviction_quality CSV where the first column embeds
    a comma within parentheses, e.g. "(hout, wout)" and "(0, 0)".

    Primary path: manual regex parsing per line.
    Fallback: DictReader/positional parsing if regex fails entirely.
    """
    rows: List[Row] = []

    # Primary: line regex parsing to handle '(h,w), ...' values
    pat = re.compile(
        r"^\s*\(\s*(\-?\d+)\s*,\s*(\-?\d+)\s*\)\s*,\s*"  # (hout, wout),
        r"(\d+)\s*,\s*"                                          # tile_id
        r"(\d+)\s*,\s*"                                          # timestep
        r"(\d+)\s*,\s*"                                          # evict total
        r"(\d+)\s*,\s*"                                          # evict bad
        r"(\d+)\s*,\s*"                                          # evict good
        r"([0-9eE+\-.]+)\s*,\s*"                                 # bad rate
        r"([0-9eE+\-.]+)\s*$"                                     # good rate
    )
    with open(csv_path, "r", newline="") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.lower().startswith("output_pos"):
                continue
            m = pat.match(s)
            if not m:
                continue
            hout = int(m.group(1)); wout = int(m.group(2))
            tile = int(m.group(3)); timestep = int(m.group(4))
            ev_total = int(m.group(5)); ev_bad = int(m.group(6)); ev_good = int(m.group(7))
            bad_rate = float(m.group(8)); good_rate = float(m.group(9))
            rows.append(Row(hout, wout, tile, timestep, ev_total, ev_bad, ev_good, bad_rate, good_rate))
    if rows:
        return rows

    # Fallbacks: try a DictReader and then positional parsing
    def norm(s: str) -> str:
        s = s.strip().lower()
        s = re.sub(r"[\s_]+", "", s)
        s = s.replace("(", "").replace(")", "").replace(",", "").replace("-", "")
        return s

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        fn = reader.fieldnames or []
        norm_map: Dict[str, str] = {norm(k): k for k in fn}

        key_pos = None
        for cand in ("outputposhoutwout", "outputpos", "output_poshoutwout"):
            if cand in norm_map:
                key_pos = norm_map[cand]
                break
        key_tile = norm_map.get("tileid") or norm_map.get("tile")
        key_ts = norm_map.get("timestep") or norm_map.get("timesteps") or norm_map.get("ts")
        key_tot = norm_map.get("evicttotal") or norm_map.get("evict_total")
        key_bad = norm_map.get("evictbad") or norm_map.get("evict_bad")
        key_good = norm_map.get("evictgood") or norm_map.get("evict_good")
        key_badr = norm_map.get("badrate") or norm_map.get("bad_rate")
        key_goodr = norm_map.get("goodrate") or norm_map.get("good_rate")

        used_dictreader = all([key_pos, key_tile, key_ts, key_tot, key_bad, key_good])

        if used_dictreader:
            for raw in reader:
                try:
                    pos = str(raw.get(key_pos, "")).strip()
                    m = re.match(r"\((\-?\d+)\s*,\s*(\-?\d+)\)", pos)
                    if not m:
                        continue
                    hout = int(m.group(1)); wout = int(m.group(2))
                    tile = int(str(raw.get(key_tile, "")).strip())
                    timestep = int(str(raw.get(key_ts, "")).strip())
                    ev_total = int(str(raw.get(key_tot, "0")).strip() or 0)
                    ev_bad = int(str(raw.get(key_bad, "0")).strip() or 0)
                    ev_good = int(str(raw.get(key_good, "0")).strip() or 0)
                    bad_rate = float(str(raw.get(key_badr, "0")).strip() or 0.0) if key_badr else 0.0
                    good_rate = float(str(raw.get(key_goodr, "0")).strip() or 0.0) if key_goodr else 0.0
                    rows.append(Row(hout, wout, tile, timestep, ev_total, ev_bad, ev_good, bad_rate, good_rate))
                except Exception:
                    continue
        else:
            f.seek(0)
            reader2 = csv.reader(f, skipinitialspace=True)
            for i, cols in enumerate(reader2):
                if not cols:
                    continue
                if i == 0 and not (cols[0].startswith("(") or cols[0].lower().startswith("output_pos")):
                    continue
                try:
                    if cols[0].startswith("(") and len(cols) >= 2 and cols[1].endswith(")"):
                        pos = cols[0] + "," + cols[1]
                        rest = cols[2:]
                    else:
                        pos = cols[0]
                        rest = cols[1:]
                    m = re.match(r"\((\-?\d+)\s*,\s*(\-?\d+)\)", pos.strip())
                    if not m or len(rest) < 7:
                        continue
                    hout = int(m.group(1)); wout = int(m.group(2))
                    tile = int(rest[0].strip())
                    timestep = int(rest[1].strip())
                    ev_total = int(rest[2].strip())
                    ev_bad = int(rest[3].strip())
                    ev_good = int(rest[4].strip())
                    bad_rate = float(rest[5].strip())
                    good_rate = float(rest[6].strip())
                    rows.append(Row(hout, wout, tile, timestep, ev_total, ev_bad, ev_good, bad_rate, good_rate))
                except Exception:
                    continue
    return rows


def _unique_sorted(values: Iterable[int]) -> List[int]:
    return sorted(set(int(v) for v in values))


def _group_by_position(rows: Sequence[Row]) -> Dict[Tuple[int, int], List[Row]]:
    out: Dict[Tuple[int, int], List[Row]] = {}
    for r in rows:
        out.setdefault((r.hout, r.wout), []).append(r)
    for k in out:
        out[k].sort(key=lambda r: (r.timestep, r.tile))
    return out


def _tile_color(tile_id: int) -> str:
    if tile_id < len(TILE_COLORS):
        return TILE_COLORS[tile_id]
    # Fallback to tab10 if more tiles appear unexpectedly
    try:
        import matplotlib as mpl
        return mpl.colormaps["tab10"](tile_id % 10)
    except Exception:
        return TILE_COLORS[tile_id % len(TILE_COLORS)]


def _make_tile_legend_handles(tiles: Sequence[int]) -> List[mpatches.Patch]:
    handles: List[mpatches.Patch] = []
    for t in sorted(set(int(x) for x in tiles)):
        handles.append(mpatches.Patch(facecolor=_tile_color(t), edgecolor="black", label=f"Tile {t}"))
    return handles


def _make_component_legend_handles() -> List[mpatches.Patch]:
    return [
        mpatches.Patch(facecolor="white", edgecolor="black", hatch=HATCH_GOOD, label="Good"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch=HATCH_BAD, label="Bad"),
    ]


def _clustered_stacked_bar(ax: plt.Axes,
                           group: List[Row],
                           view: str = "rates",
                           title: str | None = None,
                           show_legends: bool = True,
                           ylimit: Tuple[float, float] | None = None) -> None:
    """Draw clustered (per timestep) stacked bars on the given axes.

    One bar per tile for each timestep; supports 1, 2, or 4 tiles.
    view: 'counts' or 'rates'
    ylimit: if provided, sets (ymin,ymax); for 'rates' we default to (0,1).
    """
    timesteps = _unique_sorted(r.timestep for r in group)
    tile_ids = _unique_sorted(r.tile for r in group)
    n_tiles = max(1, len(tile_ids))

    # Prepare arrays [tile_index][timestep_index]
    ts_index = {t: i for i, t in enumerate(timesteps)}
    tile_index = {t: i for i, t in enumerate(tile_ids)}
    good = np.zeros((n_tiles, len(timesteps)), dtype=float)
    bad = np.zeros_like(good)
    for r in group:
        ti = tile_index[r.tile]
        tj = ts_index[r.timestep]
        if view == "counts":
            good[ti, tj] += float(r.ev_good)
            bad[ti, tj] += float(r.ev_bad)
        else:
            # Compute from counts-derived rates for stability
            good[ti, tj] += float(r.good_rate_calc)
            bad[ti, tj] += float(r.bad_rate_calc)

    n_ts = len(timesteps)
    x = np.arange(n_ts, dtype=float)
    width = 0.8 / n_tiles
    offsets = (np.arange(n_tiles) - (n_tiles - 1) / 2.0) * width

    # Bars: good on bottom, bad on top
    for k, tile_id in enumerate(tile_ids):
        xk = x + offsets[k]
        gk = good[k, :]
        bk = bad[k, :]
        color = _tile_color(tile_id)
        ax.bar(xk, gk, width=width, color=color, edgecolor="black", hatch=HATCH_GOOD, zorder=3)
        ax.bar(xk, bk, width=width, bottom=gk, color=color, edgecolor="black", hatch=HATCH_BAD, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in timesteps])
    ax.set_xlabel("Timestep")
    if view == "rates":
        ax.set_ylabel("Rate")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_ylim(0.0, 1.0)
    else:
        ax.set_ylabel("Count")
        if ylimit is not None:
            ax.set_ylim(*ylimit)

    ax.grid(True, axis="y", alpha=0.25, linestyle=":", zorder=0)
    if title:
        ax.set_title(title)

    if show_legends:
        tile_handles = _make_tile_legend_handles(tile_ids)
        comp_handles = _make_component_legend_handles()
        leg1 = ax.legend(handles=tile_handles, title="Tiles", loc="upper left", bbox_to_anchor=(1.02, 1.0))
        ax.add_artist(leg1)
        ax.legend(handles=comp_handles, title="Components", loc="upper left", bbox_to_anchor=(1.02, 0.65))


def _grid_for_per_page(n: int) -> Tuple[int, int]:
    best = (1, n)
    best_score = 1e9
    for rows in range(1, n + 1):
        cols = math.ceil(n / rows)
        score = abs(rows - cols)
        if rows <= cols and score < best_score and rows * cols >= n:
            best = (rows, cols)
            best_score = score
    return best


def plot_per_position(rows: Sequence[Row], outdir: str, do_counts: bool = False, do_rates: bool = True) -> None:
    groups = _group_by_position(rows)
    for (h, w), grp in sorted(groups.items()):
        title = f"(hout={h}, wout={w})"
        if do_counts:
            out_counts = os.path.join(outdir, f"{h}_{w}_evq_counts.pdf")
            plt.figure(figsize=(7, 5))
            _clustered_stacked_bar(plt.gca(), grp, view="counts", title=title, show_legends=True)
            plt.subplots_adjust(right=0.80, top=0.92)
            _ensure_dir(os.path.dirname(out_counts))
            plt.savefig(out_counts)
            plt.close()
        if do_rates:
            out_rates = os.path.join(outdir, f"{h}_{w}_evq_rates.pdf")
            plt.figure(figsize=(7, 5))
            _clustered_stacked_bar(plt.gca(), grp, view="rates", title=title, show_legends=True)
            plt.subplots_adjust(right=0.80, top=0.92)
            _ensure_dir(os.path.dirname(out_rates))
            plt.savefig(out_rates)
            plt.close()


def plot_small_multiples(rows: Sequence[Row], outdir: str, view: str = "rates", per_page: int = 8) -> None:
    assert view in ("counts", "rates")
    groups = _group_by_position(rows)
    positions = sorted(groups.keys())
    n = len(positions)
    if n == 0:
        return

    def page_ylim(pos_slice: Sequence[Tuple[int, int]]) -> Tuple[float, float] | None:
        if view == "rates":
            return (0.0, 1.0)
        ymax = 0.0
        for p in pos_slice:
            for r in groups[p]:
                ymax = max(ymax, float(r.total))
        if ymax <= 0:
            return (0.0, 1.0)
        return (0.0, ymax * 1.1)

    out_name = os.path.join(outdir, f"small_multiples_evq_{view}.pdf")
    _ensure_dir(os.path.dirname(out_name))
    with PdfPages(out_name) as pdf:
        i = 0
        while i < n:
            chunk = positions[i:i + per_page]
            rows_n, cols_n = _grid_for_per_page(len(chunk))
            fig_w = max(10, cols_n * 3.2) + 2.2
            fig_h = max(6, rows_n * 2.8)
            fig, axes = plt.subplots(rows_n, cols_n, figsize=(fig_w, fig_h), squeeze=False)
            fig.subplots_adjust(right=0.80, top=0.90)
            ylim = page_ylim(chunk)

            for idx, pos in enumerate(chunk):
                r_i = idx // cols_n
                c_i = idx % cols_n
                ax = axes[r_i][c_i]
                grp = groups[pos]
                _clustered_stacked_bar(ax, grp, view=view, title=f"(h={pos[0]}, w={pos[1]})", show_legends=False, ylimit=ylim)
                ax.label_outer()

            # Figure-level legends on the right
            tiles_for_chunk: List[int] = sorted({rr.tile for pos in chunk for rr in groups[pos]})
            tile_handles = _make_tile_legend_handles(tiles_for_chunk)
            comp_handles = _make_component_legend_handles()
            fig.legend(handles=tile_handles, title="Tiles", loc="upper left", bbox_to_anchor=(0.82, 0.98))
            fig.legend(handles=comp_handles, title="Components", loc="upper left", bbox_to_anchor=(0.82, 0.70))

            pdf.savefig(fig)
            plt.close(fig)
            i += per_page


def plot_overall_per_timestep(rows: Sequence[Row], outdir: str, do_counts: bool = False, do_rates: bool = True) -> None:
    # Aggregate across all positions and tiles
    by_ts: Dict[int, Tuple[int, int]] = {}  # t -> (good, bad)
    for r in rows:
        good, bad = by_ts.get(r.timestep, (0, 0))
        by_ts[r.timestep] = (good + r.ev_good, bad + r.ev_bad)

    timesteps = sorted(by_ts.keys())
    goods = np.array([by_ts[t][0] for t in timesteps], dtype=float)
    bads = np.array([by_ts[t][1] for t in timesteps], dtype=float)
    totals = goods + bads

    x = np.arange(len(timesteps))

    if do_counts:
        out_counts = os.path.join(outdir, "overall_per_timestep_evq_counts.pdf")
        _ensure_dir(os.path.dirname(out_counts))
        plt.figure(figsize=(7, 5))
        plt.bar(x, goods, color=COLOR_GOOD, edgecolor="black", hatch=HATCH_GOOD, label="Good", zorder=3)
        plt.bar(x, bads, bottom=goods, color=COLOR_BAD, edgecolor="black", hatch=HATCH_BAD, label="Bad", zorder=3)
        plt.xticks(x, [str(t) for t in timesteps])
        plt.xlabel("Timestep")
        plt.ylabel("Count")
        plt.title("Overall Good/Bad Evictions per Timestep (Counts)")
        plt.grid(True, axis="y", alpha=0.25, linestyle=":", zorder=0)
        plt.legend(title="Components", loc="upper left", bbox_to_anchor=(1.01, 1.0))
        plt.subplots_adjust(right=0.80, top=0.92)
        plt.savefig(out_counts)
        plt.close()

    if do_rates:
        out_rates = os.path.join(outdir, "overall_per_timestep_evq_rates.pdf")
        with np.errstate(divide='ignore', invalid='ignore'):
            good_rate = np.divide(goods, totals, out=np.zeros_like(goods), where=(totals > 0))
            bad_rate = np.divide(bads, totals, out=np.zeros_like(bads), where=(totals > 0))
        plt.figure(figsize=(7, 5))
        plt.bar(x, good_rate, color=COLOR_GOOD, edgecolor="black", hatch=HATCH_GOOD, label="Good", zorder=3)
        plt.bar(x, bad_rate, bottom=good_rate, color=COLOR_BAD, edgecolor="black", hatch=HATCH_BAD, label="Bad", zorder=3)
        plt.xticks(x, [str(t) for t in timesteps])
        plt.xlabel("Timestep")
        plt.ylabel("Rate")
        plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        plt.ylim(0.0, 1.0)
        plt.title("Overall Good/Bad Evictions per Timestep (Rates)")
        plt.grid(True, axis="y", alpha=0.25, linestyle=":", zorder=0)
        plt.legend(title="Components", loc="upper left", bbox_to_anchor=(1.01, 1.0))
        plt.subplots_adjust(right=0.80, top=0.92)
        plt.savefig(out_rates)
        plt.close()


def _expand_plots_list(plots: List[str]) -> List[str]:
    s = set(p.strip().lower() for p in plots if p)
    if "all" in s:
        return [
            "perpos_counts", "perpos_rates",
            "small_counts", "small_rates",
            "overall_counts", "overall_rates",
        ]
    if "perpos" in s:
        s.update(["perpos_counts", "perpos_rates"])
    if "small" in s or "small_multiples" in s:
        s.update(["small_counts", "small_rates"])
    if "overall" in s:
        s.update(["overall_counts", "overall_rates"])
    order = [
        "perpos_counts", "perpos_rates",
        "small_counts", "small_rates",
        "overall_counts", "overall_rates",
    ]
    return [k for k in order if k in s]


def process(csv_path: str, outdir: str, per_page: int, plots: List[str]) -> None:
    rows = read_evq_csv(csv_path)
    if not rows:
        print(f"[warn] No data rows parsed from {csv_path}")
        return

    _ensure_dir(outdir)
    wanted = _expand_plots_list(plots)

    if ("perpos_counts" in wanted) or ("perpos_rates" in wanted):
        print("Generating per-position eviction-quality charts...")
        plot_per_position(rows, outdir,
                          do_counts=("perpos_counts" in wanted),
                          do_rates=("perpos_rates" in wanted))

    if "small_counts" in wanted:
        print("Generating small multiples (counts)...")
        plot_small_multiples(rows, outdir, view="counts", per_page=per_page)
    if "small_rates" in wanted:
        print("Generating small multiples (rates)...")
        plot_small_multiples(rows, outdir, view="rates", per_page=per_page)

    if ("overall_counts" in wanted) or ("overall_rates" in wanted):
        print("Generating overall per-timestep stacked bars...")
        plot_overall_per_timestep(rows, outdir,
                                  do_counts=("overall_counts" in wanted),
                                  do_rates=("overall_rates" in wanted))

    print(f"Done. Outputs written under: {outdir}")


def main(argv: List[str]) -> int:
    default_csv, default_outdir = _default_paths()
    p = argparse.ArgumentParser(description="Plot eviction quality (good/bad) over timesteps and tiles.")
    p.add_argument("--csv", default=default_csv, help="Path to input eviction_quality_*.csv")
    p.add_argument("--outdir", default=None, help="Output directory (default: <csv_stem>/)")
    p.add_argument("--per-page", type=int, default=8, help="Positions per page for small multiples")
    p.add_argument(
        "--plots",
        type=str,
        default="small_counts,overall_counts",
        help=(
            "Comma-separated list of which plots to generate. Options: "
            "all, perpos, perpos_counts, perpos_rates, "
            "small, small_counts, small_rates, overall, overall_counts, overall_rates"
        ),
    )
    args = p.parse_args(argv)

    csv_path = args.csv
    if not os.path.isfile(csv_path):
        print(f"Input CSV not found: {csv_path}", file=sys.stderr)
        return 2
    outdir = args.outdir or os.path.splitext(csv_path)[0]

    plots = [p for p in (args.plots or "").split(',') if p]
    process(csv_path, outdir, per_page=max(1, int(args.per_page)), plots=plots)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
