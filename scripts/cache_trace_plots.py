#!/usr/bin/env python3
"""
Visualize cache trace hit/miss distributions over timesteps and tiles.

Inputs:
  - A CSV with header like:
      output_pos(hout, wout), tile_id, timesteps, hit count, miss count, cold miss count, conflict miss count

Behavior:
  - Facet by output position (hout,wout): for each timestep draw a cluster of
    two bars (tile 0 and tile 1). Each bar is stacked: hit (lower) + miss (upper).
  - Generate both counts and percent views. Percent shows composition (rates),
    counts shows magnitude.
  - Use one legend for tiles (colors) and a separate legend for stack
    components (hatches for hit/miss).
  - With many positions, generate small multiples (a few positions per page)
    and add heatmaps for bird’s‑eye summaries across positions and timesteps.
  - Also generate stacked bars aggregated across all positions and tiles per
    timestep (overall hit/miss counts and rates per timestep).

Outputs:
  - By default, for CSV path
      stats/repo4/vgg16/layer5/cache_traces/lru/144KB_4ways_0prefetches.csv
    outputs are written to
      stats/repo4/vgg16/layer5/cache_traces/lru/144KB_4ways_0prefetches/
    with filenames like:
      <hout>_<wout>_counts.pdf, <hout>_<wout>_rates.pdf,
      small_multiples_counts.pdf, small_multiples_rates.pdf,
      heatmaps_hit_rate.pdf, heatmaps_miss_rate.pdf,
      overall_per_timestep_counts.pdf, overall_per_timestep_rates.pdf

Usage examples:
  - python3 scripts/cache_trace_plots.py
  - python3 scripts/cache_trace_plots.py --csv /.../144KB_4ways_0prefetches.csv
  - python3 scripts/cache_trace_plots.py --per-page 8
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


# Matplotlib color palette (aligned with other scripts in this repo)
COLOR_TILE_0 = "#4C78A8"  # blue
COLOR_TILE_1 = "#F58518"  # orange
HATCH_HIT = "////"
HATCH_MISS = ".."


@dataclass
class Row:
    hout: int
    wout: int
    tile: int
    timestep: int
    hit: int
    miss: int
    cold_miss: int
    conflict_miss: int

    @property
    def total(self) -> int:
        return int(self.hit) + int(self.miss)

    @property
    def hit_rate(self) -> float:
        t = self.total
        return (self.hit / t) if t > 0 else 0.0

    @property
    def miss_rate(self) -> float:
        t = self.total
        return (self.miss / t) if t > 0 else 0.0


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
        "144KB_4ways_0prefetches.csv",
    )
    outdir = os.path.splitext(csv_path)[0]
    return csv_path, outdir


def read_cache_csv(csv_path: str) -> List[Row]:
    """Robust reader for the special CSV where the first column and header
    contain a comma within parentheses, e.g. "(hout, wout)" and "(0, 0)".

    Primary path: manual regex parsing per line.
    Fallback: DictReader/positional parsing if regex fails entirely.
    """
    rows: List[Row] = []

    # 1) Primary: manual line regex parsing to handle '(h,w), ...' values
    pat = re.compile(
        r"^\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*"  # (hout, wout),
        r"(\d+)\s*,\s*"                                   # tile
        r"(\d+)\s*,\s*"                                   # timestep
        r"(\d+)\s*,\s*"                                   # hit
        r"(\d+)\s*,\s*"                                   # miss
        r"(\d+)\s*,\s*"                                   # cold miss
        r"(\d+)\s*$"                                       # conflict miss
    )
    with open(csv_path, "r", newline="") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # Skip header line if present
            if s.lower().startswith("output_pos"):
                continue
            m = pat.match(s)
            if not m:
                continue
            hout, wout, tile, ts, hit, miss, cold, conf = map(int, m.groups())
            rows.append(Row(hout, wout, tile, ts, hit, miss, cold, conf))
    if rows:
        return rows

    def norm(s: str) -> str:
        s = s.strip().lower()
        # Remove spaces, underscores, and most punctuation to match loosely
        s = re.sub(r"[\s_]+", "", s)
        s = s.replace("(", "").replace(")", "").replace(",", "").replace("-", "")
        return s

    with open(csv_path, "r", newline="") as f:
        # First, try DictReader with normalized header matching
        reader = csv.DictReader(f, skipinitialspace=True)
        fn = reader.fieldnames or []
        norm_map: Dict[str, str] = {norm(k): k for k in fn}

        key_pos = None
        for cand in ("outputposhoutwout", "outputposhout,wout", "outputpos", "output_poshoutwout"):
            if cand in norm_map:
                key_pos = norm_map[cand]
                break
        key_tile = norm_map.get("tileid") or norm_map.get("tile")
        key_ts = norm_map.get("timesteps") or norm_map.get("timestep") or norm_map.get("ts")
        key_hit = norm_map.get("hitcount") or norm_map.get("hit")
        key_miss = norm_map.get("misscount") or norm_map.get("miss")
        key_cold = norm_map.get("coldmisscount") or norm_map.get("coldmiss") or norm_map.get("cold")
        key_conf = norm_map.get("conflictmisscount") or norm_map.get("conflictmiss") or norm_map.get("conflict")

        used_dictreader = all([key_pos, key_tile, key_ts, key_hit, key_miss])

        if used_dictreader:
            for raw in reader:
                try:
                    pos = str(raw.get(key_pos, "")).strip()
                    m = re.match(r"\((\d+)\s*,\s*(\d+)\)", pos)
                    if not m:
                        # Not a data line
                        continue
                    hout = int(m.group(1))
                    wout = int(m.group(2))
                    tile = int(str(raw.get(key_tile, "")).strip())
                    timestep = int(str(raw.get(key_ts, "")).strip())
                    hit = int(str(raw.get(key_hit, "0")).strip() or 0)
                    miss = int(str(raw.get(key_miss, "0")).strip() or 0)
                    cold_miss = int(str(raw.get(key_cold, "0")).strip() or 0) if key_cold else 0
                    conflict_miss = int(str(raw.get(key_conf, "0")).strip() or 0) if key_conf else 0
                    rows.append(Row(hout, wout, tile, timestep, hit, miss, cold_miss, conflict_miss))
                except Exception:
                    continue
        else:
            # Fallback: positional parse via csv.reader
            f.seek(0)
            reader2 = csv.reader(f, skipinitialspace=True)
            for i, cols in enumerate(reader2):
                if not cols:
                    continue
                # Skip header if it looks like it
                if i == 0 and not (cols[0].startswith("(") or cols[0].lower().startswith("output_pos")):
                    continue
                try:
                    # Reconstruct the first two tokens if the position was split by the comma
                    if cols[0].startswith("(") and len(cols) >= 2 and cols[1].endswith(")"):
                        pos = cols[0] + "," + cols[1]
                        rest = cols[2:]
                    else:
                        pos = cols[0]
                        rest = cols[1:]
                    m = re.match(r"\((\d+)\s*,\s*(\d+)\)", pos.strip())
                    if not m or len(rest) < 6:
                        continue
                    hout = int(m.group(1))
                    wout = int(m.group(2))
                    tile = int(rest[0].strip())
                    timestep = int(rest[1].strip())
                    hit = int(rest[2].strip())
                    miss = int(rest[3].strip())
                    cold_miss = int(rest[4].strip())
                    conflict_miss = int(rest[5].strip())
                    rows.append(Row(hout, wout, tile, timestep, hit, miss, cold_miss, conflict_miss))
                except Exception:
                    continue
    return rows


def _unique_sorted(values: Iterable[int]) -> List[int]:
    return sorted(set(int(v) for v in values))


def _group_by_position(rows: Sequence[Row]) -> Dict[Tuple[int, int], List[Row]]:
    out: Dict[Tuple[int, int], List[Row]] = {}
    for r in rows:
        out.setdefault((r.hout, r.wout), []).append(r)
    # Sort each group's rows by (timestep, tile)
    for k in out:
        out[k].sort(key=lambda r: (r.timestep, r.tile))
    return out


def _make_tile_legend_handles() -> List[mpatches.Patch]:
    return [
        mpatches.Patch(facecolor=COLOR_TILE_0, edgecolor="black", label="Tile 0"),
        mpatches.Patch(facecolor=COLOR_TILE_1, edgecolor="black", label="Tile 1"),
    ]


def _make_component_legend_handles() -> List[mpatches.Patch]:
    return [
        mpatches.Patch(facecolor="white", edgecolor="black", hatch=HATCH_HIT, label="Hit"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch=HATCH_MISS, label="Miss"),
    ]


def _clustered_stacked_bar(ax: plt.Axes,
                           group: List[Row],
                           view: str = "counts",
                           title: str | None = None,
                           show_legends: bool = True,
                           ylimit: Tuple[float, float] | None = None) -> None:
    """Draw clustered (per timestep) stacked bars (tile 0/1) on the given axes.

    view: 'counts' or 'rates'
    ylimit: if provided, sets (ymin,ymax); for 'rates' we default to (0,1).
    """
    # Collect by timestep then tile
    timesteps = _unique_sorted(r.timestep for r in group)
    # Expect tile ids 0 and 1; fill missing with zeros to keep layout stable
    tiles = [0, 1]
    color_map = {0: COLOR_TILE_0, 1: COLOR_TILE_1}

    # Build arrays in timestep order
    hit_vals: Dict[int, List[float]] = {t: [0.0, 0.0] for t in timesteps}
    miss_vals: Dict[int, List[float]] = {t: [0.0, 0.0] for t in timesteps}
    lookup: Dict[Tuple[int, int], Row] = {(r.timestep, r.tile): r for r in group}
    for t in timesteps:
        for ti in tiles:
            r = lookup.get((t, ti))
            if r is None:
                hit_vals[t][ti] = 0.0
                miss_vals[t][ti] = 0.0
            else:
                if view == "counts":
                    hit_vals[t][ti] = float(r.hit)
                    miss_vals[t][ti] = float(r.miss)
                else:  # rates
                    hit_vals[t][ti] = r.hit_rate
                    miss_vals[t][ti] = r.miss_rate

    # Layout: for each timestep, two bars (tile 0 and 1) side-by-side
    n_ts = len(timesteps)
    indices = np.arange(n_ts, dtype=float)
    width = 0.35  # per-bar width
    offset = width / 2

    # Prepare bar positions for each tile
    x0 = indices - offset
    x1 = indices + offset

    # Build arrays of heights in timestep order
    hit0 = [hit_vals[t][0] for t in timesteps]
    miss0 = [miss_vals[t][0] for t in timesteps]
    hit1 = [hit_vals[t][1] for t in timesteps]
    miss1 = [miss_vals[t][1] for t in timesteps]

    # Draw bars (stacked hits + misses)
    ax.bar(x0, hit0, width=width, color=COLOR_TILE_0, edgecolor="black", hatch=HATCH_HIT, zorder=3)
    ax.bar(x0, miss0, width=width, bottom=hit0, color=COLOR_TILE_0, edgecolor="black", hatch=HATCH_MISS, zorder=3)

    ax.bar(x1, hit1, width=width, color=COLOR_TILE_1, edgecolor="black", hatch=HATCH_HIT, zorder=3)
    ax.bar(x1, miss1, width=width, bottom=hit1, color=COLOR_TILE_1, edgecolor="black", hatch=HATCH_MISS, zorder=3)

    ax.set_xticks(indices)
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
        tile_handles = _make_tile_legend_handles()
        comp_handles = _make_component_legend_handles()
        # Place legends outside to avoid clutter
        leg1 = ax.legend(handles=tile_handles, title="Tiles", loc="upper left", bbox_to_anchor=(1.02, 1.0))
        ax.add_artist(leg1)
        ax.legend(handles=comp_handles, title="Components", loc="upper left", bbox_to_anchor=(1.02, 0.65))


def _grid_for_per_page(n: int) -> Tuple[int, int]:
    # Choose a near-square grid but prefer 2x4, 3x4, 4x4 patterns
    # Try to keep rows <= cols
    best = (1, n)
    best_score = 1e9
    for rows in range(1, n + 1):
        cols = math.ceil(n / rows)
        score = abs(rows - cols)
        if rows <= cols and score < best_score and rows * cols >= n:
            best = (rows, cols)
            best_score = score
    return best


def plot_per_position(rows: Sequence[Row], outdir: str, do_counts: bool = True, do_rates: bool = True) -> None:
    groups = _group_by_position(rows)
    for (h, w), grp in sorted(groups.items()):
        title = f"(hout={h}, wout={w})"
        # Counts view
        if do_counts:
            out_counts = os.path.join(outdir, f"{h}_{w}_counts.pdf")
            plt.figure(figsize=(7, 5))
            _clustered_stacked_bar(plt.gca(), grp, view="counts", title=title, show_legends=True)
            plt.subplots_adjust(right=0.80, top=0.92)
            _ensure_dir(os.path.dirname(out_counts))
            plt.savefig(out_counts)
            plt.close()

        # Rates view
        if do_rates:
            out_rates = os.path.join(outdir, f"{h}_{w}_rates.pdf")
            plt.figure(figsize=(7, 5))
            _clustered_stacked_bar(plt.gca(), grp, view="rates", title=title, show_legends=True)
            plt.subplots_adjust(right=0.80, top=0.92)
            _ensure_dir(os.path.dirname(out_rates))
            plt.savefig(out_rates)
            plt.close()


def plot_small_multiples(rows: Sequence[Row], outdir: str, view: str = "counts", per_page: int = 8) -> None:
    assert view in ("counts", "rates")
    groups = _group_by_position(rows)
    positions = sorted(groups.keys())
    n = len(positions)
    if n == 0:
        return

    # Precompute per-page y-limit for counts to keep relative scale within page
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

    # Multi-page PDF
    out_name = os.path.join(outdir, f"small_multiples_{view}.pdf")
    _ensure_dir(os.path.dirname(out_name))
    with PdfPages(out_name) as pdf:
        i = 0
        while i < n:
            chunk = positions[i:i + per_page]
            rows_n, cols_n = _grid_for_per_page(len(chunk))
            # Extra width on the right for figure-level legends
            fig_w = max(10, cols_n * 3.2) + 2.2
            fig_h = max(6, rows_n * 2.8)
            fig, axes = plt.subplots(rows_n, cols_n, figsize=(fig_w, fig_h), squeeze=False)
            # Reserve room on the right/top for legends so they don't overlap bars
            fig.subplots_adjust(right=0.80, top=0.90)
            ylim = page_ylim(chunk)

            for idx, pos in enumerate(chunk):
                r = idx // cols_n
                c = idx % cols_n
                ax = axes[r][c]
                # Compact title like (0,0)
                _clustered_stacked_bar(ax, groups[pos], view=view, title=f"{pos[0]},{pos[1]}", show_legends=False, ylimit=ylim)
                # Reduce label clutter per-axes
                if view == "rates":
                    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
                if r < rows_n - 1:
                    ax.set_xlabel("")
                if c > 0:
                    ax.set_ylabel("")

            # Hide any unused subplots
            for j in range(len(chunk), rows_n * cols_n):
                r = j // cols_n
                c = j % cols_n
                axes[r][c].axis("off")

            # Shared legends for the page (placed to right, stacked vertically)
            tile_handles = _make_tile_legend_handles()
            comp_handles = _make_component_legend_handles()
            fig.legend(handles=tile_handles, title="Tiles", loc="upper left", bbox_to_anchor=(0.82, 0.98))
            fig.legend(handles=comp_handles, title="Components", loc="upper left", bbox_to_anchor=(0.82, 0.83))

            pdf.savefig(fig)
            plt.close(fig)
            i += per_page


def _aggregate_rates_by_pos_and_ts(rows: Sequence[Row]) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """Return timesteps list, hit_rate[ts,h,w], miss_rate[ts,h,w].

    Aggregated across tiles (sum hits/misses then divide by total).
    """
    timesteps = _unique_sorted(r.timestep for r in rows)
    hmax = max(r.hout for r in rows) + 1
    wmax = max(r.wout for r in rows) + 1
    hit = np.zeros((len(timesteps), hmax, wmax), dtype=float)
    miss = np.zeros_like(hit)
    total = np.zeros_like(hit)

    ts_index = {t: i for i, t in enumerate(timesteps)}
    for r in rows:
        i = ts_index[r.timestep]
        hit[i, r.hout, r.wout] += r.hit
        miss[i, r.hout, r.wout] += r.miss
        total[i, r.hout, r.wout] += r.total

    with np.errstate(divide='ignore', invalid='ignore'):
        hit_rate = np.divide(hit, total, out=np.zeros_like(hit), where=(total > 0))
        miss_rate = np.divide(miss, total, out=np.zeros_like(miss), where=(total > 0))
    return timesteps, hit_rate, miss_rate


def plot_heatmaps(rows: Sequence[Row], outdir: str) -> None:
    timesteps, hit_rate, miss_rate = _aggregate_rates_by_pos_and_ts(rows)
    if len(timesteps) == 0:
        return

    # Hit rate pages
    out_hit = os.path.join(outdir, "heatmaps_hit_rate.pdf")
    _ensure_dir(os.path.dirname(out_hit))
    fig_w = max(10, len(timesteps) * 3.2)
    fig_h = max(5, 3.2)
    fig, axes = plt.subplots(1, len(timesteps), figsize=(fig_w, fig_h), squeeze=False, constrained_layout=True)
    vmin, vmax = 0.0, 1.0
    for i, t in enumerate(timesteps):
        ax = axes[0][i]
        im = ax.imshow(hit_rate[i], vmin=vmin, vmax=vmax, cmap="viridis", origin="upper", interpolation="nearest")
        ax.set_title(f"Hit Rate (t={t})")
        ax.set_xlabel("wout")
        if i == 0:
            ax.set_ylabel("hout")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.ax.set_ylabel("rate")
    plt.savefig(out_hit)
    plt.close(fig)

    # Miss rate pages
    out_miss = os.path.join(outdir, "heatmaps_miss_rate.pdf")
    fig, axes = plt.subplots(1, len(timesteps), figsize=(fig_w, fig_h), squeeze=False, constrained_layout=True)
    for i, t in enumerate(timesteps):
        ax = axes[0][i]
        im = ax.imshow(miss_rate[i], vmin=vmin, vmax=vmax, cmap="magma", origin="upper", interpolation="nearest")
        ax.set_title(f"Miss Rate (t={t})")
        ax.set_xlabel("wout")
        if i == 0:
            ax.set_ylabel("hout")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.ax.set_ylabel("rate")
    plt.savefig(out_miss)
    plt.close(fig)


def plot_overall_per_timestep(rows: Sequence[Row], outdir: str, do_counts: bool = True, do_rates: bool = True) -> None:
    # Aggregate across all positions and tiles
    by_ts: Dict[int, Tuple[int, int]] = {}  # t -> (hit, miss)
    for r in rows:
        hit, miss = by_ts.get(r.timestep, (0, 0))
        by_ts[r.timestep] = (hit + r.hit, miss + r.miss)

    timesteps = sorted(by_ts.keys())
    hits = np.array([by_ts[t][0] for t in timesteps], dtype=float)
    misses = np.array([by_ts[t][1] for t in timesteps], dtype=float)
    totals = hits + misses

    x = np.arange(len(timesteps))

    # Counts stacked bar
    if do_counts:
        out_counts = os.path.join(outdir, "overall_per_timestep_counts.pdf")
        _ensure_dir(os.path.dirname(out_counts))
        plt.figure(figsize=(7, 5))
        plt.bar(x, hits, color="#9ACD32", edgecolor="black", hatch=HATCH_HIT, label="Hit", zorder=3)
        plt.bar(x, misses, bottom=hits, color="#CD5C5C", edgecolor="black", hatch=HATCH_MISS, label="Miss", zorder=3)
        plt.xticks(x, [str(t) for t in timesteps])
        plt.xlabel("Timestep")
        plt.ylabel("Count")
        plt.title("Overall Hit/Miss per Timestep (Counts)")
        plt.grid(True, axis="y", alpha=0.25, linestyle=":", zorder=0)
        # Place legend outside the plot on the right
        plt.legend(title="Components", loc="upper left", bbox_to_anchor=(1.01, 1.0))
        plt.subplots_adjust(right=0.80, top=0.92)
        plt.savefig(out_counts)
        plt.close()

    # Rates stacked bar
    if do_rates:
        out_rates = os.path.join(outdir, "overall_per_timestep_rates.pdf")
        with np.errstate(divide='ignore', invalid='ignore'):
            hit_rate = np.divide(hits, totals, out=np.zeros_like(hits), where=(totals > 0))
            miss_rate = np.divide(misses, totals, out=np.zeros_like(misses), where=(totals > 0))
        plt.figure(figsize=(7, 5))
        plt.bar(x, hit_rate, color="#9ACD32", edgecolor="black", hatch=HATCH_HIT, label="Hit", zorder=3)
        plt.bar(x, miss_rate, bottom=hit_rate, color="#CD5C5C", edgecolor="black", hatch=HATCH_MISS, label="Miss", zorder=3)
        plt.xticks(x, [str(t) for t in timesteps])
        plt.xlabel("Timestep")
        plt.ylabel("Rate")
        plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        plt.ylim(0.0, 1.0)
        plt.title("Overall Hit/Miss per Timestep (Rates)")
        plt.grid(True, axis="y", alpha=0.25, linestyle=":", zorder=0)
        # Place legend outside the plot on the right
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
            "heatmaps",
            "overall_counts", "overall_rates",
        ]
    # Shorthand groups
    if "perpos" in s:
        s.update(["perpos_counts", "perpos_rates"])
    if "small" in s or "small_multiples" in s:
        s.update(["small_counts", "small_rates"])
    if "overall" in s:
        s.update(["overall_counts", "overall_rates"])
    if "heatmap" in s:
        s.add("heatmaps")
    # Validate/keep order
    order = [
        "perpos_counts", "perpos_rates",
        "small_counts", "small_rates",
        "heatmaps",
        "overall_counts", "overall_rates",
    ]
    return [k for k in order if k in s]


def process(csv_path: str, outdir: str, per_page: int, plots: List[str]) -> None:
    rows = read_cache_csv(csv_path)
    if not rows:
        print(f"[warn] No data rows parsed from {csv_path}")
        return

    _ensure_dir(outdir)

    # Expand plots selection
    wanted = _expand_plots_list(plots)

    # Per-position PDFs
    if ("perpos_counts" in wanted) or ("perpos_rates" in wanted):
        print("Generating per-position charts...")
        plot_per_position(rows, outdir,
                          do_counts=("perpos_counts" in wanted),
                          do_rates=("perpos_rates" in wanted))

    # Small multiples
    if "small_counts" in wanted:
        print("Generating small multiples (counts)...")
        plot_small_multiples(rows, outdir, view="counts", per_page=per_page)
    if "small_rates" in wanted:
        print("Generating small multiples (rates)...")
        plot_small_multiples(rows, outdir, view="rates", per_page=per_page)

    # Heatmaps
    if "heatmaps" in wanted:
        print("Generating heatmap summaries (rates)...")
        plot_heatmaps(rows, outdir)

    # Overall per timestep (across positions and tiles)
    if ("overall_counts" in wanted) or ("overall_rates" in wanted):
        print("Generating overall per-timestep stacked bars...")
        plot_overall_per_timestep(rows, outdir,
                                  do_counts=("overall_counts" in wanted),
                                  do_rates=("overall_rates" in wanted))

    print(f"Done. Outputs written under: {outdir}")


def main(argv: List[str]) -> int:
    default_csv, default_outdir = _default_paths()
    p = argparse.ArgumentParser(description="Plot cache trace hit/miss distributions over timesteps and tiles.")
    p.add_argument("--csv", default=default_csv, help="Path to input CSV")
    p.add_argument("--outdir", default=None, help="Output directory (default: <csv_stem>/)")
    p.add_argument("--per-page", type=int, default=8, help="Positions per page for small multiples")
    p.add_argument(
        "--plots",
        type=str,
        default="all",
        help=(
            "Comma-separated list of which plots to generate. Options: "
            "all, perpos, perpos_counts, perpos_rates, "
            "small, small_counts, small_rates, heatmaps, overall, overall_counts, overall_rates"
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
