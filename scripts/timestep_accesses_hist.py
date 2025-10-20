#!/usr/bin/env python3

"""
Compute distribution histograms from a timestep_accesses CSV.

Input CSV format (one row per output spine):
    output_spine_id,t0,t1,t2,t3

This script stacks t0..t3 into long form and computes, for each timestep,
the frequency of each observed access value and its share. It can print the
result to stdout or write a distribution CSV next to the input file.

Output CSV columns:
    timestep,value,count,share[,source_spine]

Notes:
- "share" is the fraction of rows within the timestep (count / N for that t).
- If the input path contains .../scoreboard/<spine>/..., the <spine> token is
  captured and included as "source_spine" for traceability.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


TIMESTEP_COLS = ("t0", "t1", "t2", "t3")


@dataclass
class DistributionRow:
    timestep: str
    value: int
    count: int
    share: float
    source_spine: str | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute distribution histograms for timestep accesses (t0..t3) from "
            "timestep_accesses_*.csv files."
        )
    )
    p.add_argument(
        "input",
        nargs="?",
        default=".",
        help="Path to a CSV file or directory to scan (default: current dir).",
    )
    p.add_argument(
        "--pattern",
        default="timestep_accesses_*.csv",
        help="Glob pattern to match when input is a directory (default: %(default)s).",
    )
    p.add_argument(
        "--write",
        action="store_true",
        help=(
            "Write a distribution CSV next to each input file. Without this, "
            "results are printed to stdout."
        ),
    )
    p.add_argument(
        "--include-spine",
        action="store_true",
        help=(
            "Include a source_spine column when a scoreboard/<spine>/ segment is found "
            "in the input path."
        ),
    )
    p.add_argument(
        "--combine-all",
        action="store_true",
        help=(
            "Also include a combined 'all' timestep where t0..t3 are merged before "
            "computing the distribution."
        ),
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help=(
            "Generate PNG histogram plots for t0..t3 (and 'all' if --combine-all)."
        ),
    )
    p.add_argument(
        "--tail-share",
        type=float,
        default=1.0,
        help=(
            "Keep bins until cumulative share reaches this fraction and drop the tail. "
            "Use 1.0 to disable (default)."
        ),
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved figures when --plot is used (default: %(default)s).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        help=(
            "Optional output directory for written CSV/PNG files. If omitted, files "
            "are written next to each input CSV."
        ),
    )
    return p.parse_args()


def find_csvs(root_or_file: Path, pattern: str) -> List[Path]:
    if root_or_file.is_file():
        return [root_or_file]
    if not root_or_file.exists():
        return []
    return sorted(root_or_file.rglob(pattern))


def read_timestep_values(csv_path: Path) -> Dict[str, List[int]]:
    values: Dict[str, List[int]] = {k: [] for k in TIMESTEP_COLS}
    with csv_path.open("r", encoding="ascii", newline="") as handle:
        reader = csv.DictReader(handle)
        # Expect columns: output_spine_id,t0,t1,t2,t3
        for row in reader:
            for t in TIMESTEP_COLS:
                try:
                    v = int(row[t])
                except (KeyError, ValueError):
                    # Skip malformed cells
                    continue
                values[t].append(v)
    return values


def extract_spine_from_path(csv_path: Path) -> str | None:
    # Heuristic: look for .../scoreboard/<spine>/ in parents
    parts = list(csv_path.parts)
    for i, part in enumerate(parts):
        if part == "scoreboard" and i + 1 < len(parts):
            return parts[i + 1]
    return None


def compute_distribution(values: Iterable[int]) -> List[Tuple[int, int, float]]:
    vals = list(values)
    if not vals:
        return []
    total = float(len(vals))
    counter = Counter(vals)
    rows = []
    for value in sorted(counter.keys()):
        count = counter[value]
        share = count / total
        rows.append((value, count, share))
    return rows


def trim_tail(values: List[int], counts: List[int], shares: List[float], tail_share: float):
    if not values or not counts or not shares:
        return values, counts, shares, 0.0
    if not (0.0 < tail_share < 1.0):
        return values, counts, shares, 0.0
    cumulative = 0.0
    cutoff_idx = len(values) - 1
    for idx, s in enumerate(shares):
        cumulative += s
        if cumulative >= tail_share:
            cutoff_idx = idx
            break
    dropped = sum(shares[cutoff_idx + 1 :])
    return (
        values[: cutoff_idx + 1],
        counts[: cutoff_idx + 1],
        shares[: cutoff_idx + 1],
        dropped,
    )


def plot_grid_histograms(
    csv_path: Path,
    per_t_values: Dict[str, List[int]],
    source_spine: str | None,
    tail_share: float,
    dpi: int,
    out_dir: Path | None = None,
) -> Path | None:
    # Build per-timestep discrete distributions
    dist: Dict[str, Tuple[List[int], List[int], List[float]]] = {}
    nsamples: Dict[str, int] = {}
    for t in TIMESTEP_COLS:
        drows = compute_distribution(per_t_values.get(t, []))
        if not drows:
            continue
        xs = [r[0] for r in drows]
        cs = [r[1] for r in drows]
        ss = [r[2] for r in drows]
        xs, cs, ss, dropped = trim_tail(xs, cs, ss, tail_share)
        dist[t] = (xs, cs, ss)
        nsamples[t] = sum(cs)

    if not dist:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=False)
    axes = axes.flatten()
    order = list(TIMESTEP_COLS)
    for idx, t in enumerate(order):
        ax = axes[idx]
        if t in dist:
            xs, cs, _ = dist[t]
            ax.bar(xs, cs, color="#4C72B0", width=1.0, edgecolor="none")
            ax.set_title(f"{t} (N={nsamples.get(t, 0)})", fontsize=10)
            ax.set_xlabel("Access value")
            ax.set_ylabel("Count")
            ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        else:
            ax.set_visible(False)

    title = csv_path.stem
    if source_spine:
        title += f" — spine {source_spine}"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_name = re.sub(r"^timestep_accesses_", "timestep_accesses_hist_", csv_path.stem) + ".png"
    out_path = (out_dir / out_name) if out_dir else csv_path.with_name(out_name)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def plot_single_histogram(
    csv_path: Path,
    label: str,
    values: List[int],
    counts: List[int],
    shares: List[float],
    source_spine: str | None,
    tail_share: float,
    dpi: int,
    out_dir: Path | None = None,
) -> Path:
    xs, cs, ss, dropped = trim_tail(values, counts, shares, tail_share)
    if not xs:
        return csv_path
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(xs, cs, color="#4C72B0", width=1.0, edgecolor="none")
    ax.set_title(label)
    ax.set_xlabel("Access value")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    if dropped > 0:
        ax.text(
            0.99,
            0.95,
            f"Tail dropped: {dropped*100.0:.2f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
    if source_spine:
        ax.text(
            0.01,
            0.95,
            f"spine {source_spine}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
        )
    fig.tight_layout()
    out_name = (
        re.sub(r"^timestep_accesses_", "timestep_accesses_hist_", csv_path.stem)
        + f"_{label}.png"
    )
    out_path = (out_dir / out_name) if out_dir else csv_path.with_name(out_name)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def build_rows(
    per_t_values: Dict[str, List[int]],
    source_spine: str | None,
    include_combined: bool,
    include_spine: bool,
) -> List[DistributionRow]:
    out: List[DistributionRow] = []
    for t in TIMESTEP_COLS:
        dist = compute_distribution(per_t_values.get(t, []))
        for value, count, share in dist:
            out.append(
                DistributionRow(
                    timestep=t,
                    value=value,
                    count=count,
                    share=share,
                    source_spine=source_spine if include_spine else None,
                )
            )
    if include_combined:
        merged: List[int] = []
        for t in TIMESTEP_COLS:
            merged.extend(per_t_values.get(t, []))
        for value, count, share in compute_distribution(merged):
            out.append(
                DistributionRow(
                    timestep="all",
                    value=value,
                    count=count,
                    share=share,
                    source_spine=source_spine if include_spine else None,
                )
            )
    return out


def write_distribution_csv(csv_path: Path, rows: List[DistributionRow], out_dir: Path | None = None) -> Path:
    # Name like: timestep_accesses_distribution_<rest>.csv to mirror reuse_distribution naming
    stem = csv_path.stem
    suffix = csv_path.suffix
    out_stem = re.sub(r"^timestep_accesses_", "timestep_accesses_distribution_", stem)
    out_name = f"{out_stem}{suffix}"
    out_path = (out_dir / out_name) if out_dir else csv_path.with_name(out_name)
    with out_path.open("w", encoding="ascii", newline="") as handle:
        fieldnames = ["timestep", "value", "count", "share"]
        # Only include source_spine when it appears in any rows
        if any(r.source_spine is not None for r in rows):
            fieldnames.append("source_spine")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            rec = {
                "timestep": r.timestep,
                "value": r.value,
                "count": r.count,
                "share": f"{r.share:.6f}",
            }
            if "source_spine" in fieldnames:
                rec["source_spine"] = r.source_spine or ""
            writer.writerow(rec)
    return out_path


def print_distribution(rows: List[DistributionRow]) -> None:
    # Group by timestep for stable, readable output
    by_t: Dict[str, List[DistributionRow]] = {}
    for r in rows:
        by_t.setdefault(r.timestep, []).append(r)
    for t in sorted(by_t.keys(), key=lambda s: (s != "all", s)):
        print(f"# timestep={t}")
        print("value,count,share" + (",source_spine" if any(rr.source_spine for rr in by_t[t]) else ""))
        for r in by_t[t]:
            tail = f",{r.source_spine}" if r.source_spine else ""
            print(f"{r.value},{r.count},{r.share:.6f}{tail}")
        print()


def main() -> int:
    args = parse_args()
    root = Path(args.input).resolve()
    if args.out_dir is not None:
        try:
            args.out_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Failed to create out-dir {args.out_dir}: {e}", file=sys.stderr)
            return 1
    csv_paths = find_csvs(root, args.pattern)
    if not csv_paths:
        print("No matching timestep_accesses CSV files found.", file=sys.stderr)
        return 1

    rc = 0
    for csv_path in csv_paths:
        per_t = read_timestep_values(csv_path)
        source_spine = extract_spine_from_path(csv_path)
        rows = build_rows(
            per_t,
            source_spine=source_spine,
            include_combined=args.combine_all,
            include_spine=args.include_spine,
        )
        if args.write:
            out_path = write_distribution_csv(csv_path, rows, args.out_dir)
            print(f"Wrote {out_path}")
        else:
            print(f"# Source: {csv_path}")
            if args.include_spine and source_spine:
                print(f"# source_spine={source_spine}")
            print_distribution(rows)

        if args.plot:
            # Grid figure for t0..t3
            grid_path = plot_grid_histograms(
                csv_path, per_t, source_spine, args.tail_share, args.dpi, args.out_dir
            )
            if grid_path:
                print(f"Wrote {grid_path}")
            # Optional combined 'all' figure
            if args.combine_all:
                merged: List[int] = []
                for t in TIMESTEP_COLS:
                    merged.extend(per_t.get(t, []))
                drows = compute_distribution(merged)
                xs = [r[0] for r in drows]
                cs = [r[1] for r in drows]
                ss = [r[2] for r in drows]
                single_path = plot_single_histogram(
                    csv_path,
                    label="all",
                    values=xs,
                    counts=cs,
                    shares=ss,
                    source_spine=source_spine,
                    tail_share=args.tail_share,
                    dpi=args.dpi,
                    out_dir=args.out_dir,
                )
                print(f"Wrote {single_path}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
