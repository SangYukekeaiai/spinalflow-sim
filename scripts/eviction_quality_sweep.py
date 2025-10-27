#!/usr/bin/env python3
"""
Sweep a model directory for eviction_quality CSVs and generate selected figures.

CLI
  1) --model-dir  Path to a model directory, e.g. stats/repo4/vgg16/
  2) --plots      Which figures to generate (comma-separated). Examples:
                  small_counts, overall_counts, perpos_rates, overall_rates, all
     Default:     small_counts,overall_counts
  3) Skips any figure that already exists.

Notes
  - Uses scripts/eviction_quality_plots.py under the hood (imports its process()).
  - For per-position plots, checks every expected PDF (<hout>_<wout>_evq_*.pdf).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Iterable, List, Sequence, Tuple


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


sys.path.insert(0, _repo_root())
from scripts.eviction_quality_plots import (
    process as process_one,
    read_evq_csv,
    _expand_plots_list,
)


def _find_evq_csvs(model_dir: str) -> List[str]:
    csvs: List[str] = []
    for root, _dirs, files in os.walk(model_dir):
        if "cache_traces" not in root:
            continue
        for fn in files:
            if not fn.lower().endswith(".csv"):
                continue
            if not fn.lower().startswith("eviction_quality_"):
                continue
            csvs.append(os.path.join(root, fn))
    return sorted(csvs)


def _exists_all(paths: Iterable[str]) -> bool:
    return all(os.path.isfile(p) for p in paths)


def _expected_outputs(csv_path: str, plots: List[str]) -> Dict[str, List[str]]:
    outdir = os.path.splitext(csv_path)[0]
    exp: Dict[str, List[str]] = {}
    if "small_counts" in plots:
        exp["small_counts"] = [os.path.join(outdir, "small_multiples_evq_counts.pdf")]
    if "small_rates" in plots:
        exp["small_rates"] = [os.path.join(outdir, "small_multiples_evq_rates.pdf")]
    if "overall_counts" in plots:
        exp["overall_counts"] = [os.path.join(outdir, "overall_per_timestep_evq_counts.pdf")]
    if "overall_rates" in plots:
        exp["overall_rates"] = [os.path.join(outdir, "overall_per_timestep_evq_rates.pdf")]

    need_perpos = [p for p in plots if p in ("perpos_counts", "perpos_rates")]
    if need_perpos:
        rows = read_evq_csv(csv_path)
        pos_set = sorted(set((r.hout, r.wout) for r in rows))
        if "perpos_counts" in plots:
            exp["perpos_counts"] = [os.path.join(outdir, f"{h}_{w}_evq_counts.pdf") for (h, w) in pos_set]
        if "perpos_rates" in plots:
            exp["perpos_rates"] = [os.path.join(outdir, f"{h}_{w}_evq_rates.pdf") for (h, w) in pos_set]
    return exp


def process_model_dir(model_dir: str, plots: List[str], per_page: int) -> None:
    expanded = _expand_plots_list(plots)
    csvs = _find_evq_csvs(model_dir)
    if not csvs:
        print(f"[info] No eviction_quality CSVs under: {model_dir}")
        return

    print(f"[info] Found {len(csvs)} CSV(s) under {model_dir}")
    for csv_path in csvs:
        outdir = os.path.splitext(csv_path)[0]
        want_for_this: List[str] = []
        expected = _expected_outputs(csv_path, expanded)

        for key, paths in expected.items():
            if _exists_all(paths):
                print(f"[skip] {key} exists for {os.path.relpath(csv_path)}")
            else:
                want_for_this.append(key)

        if not want_for_this:
            continue

        print(f"[run] Generating {','.join(want_for_this)} for {os.path.relpath(csv_path)}")
        process_one(csv_path, outdir, per_page=per_page, plots=want_for_this)


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description="Sweep eviction_quality CSVs and generate selected figures.")
    p.add_argument(
        "--model-dir",
        default=os.path.join("stats", "repo4", "vgg16"),
        help="Model directory to scan, e.g., stats/repo4/vgg16",
    )
    p.add_argument(
        "--plots",
        type=str,
        default="small_counts,overall_counts",
        help=(
            "Which plots to generate (comma-separated). Options: "
            "all, perpos, perpos_counts, perpos_rates, small, small_counts, small_rates, overall, overall_counts, overall_rates"
        ),
    )
    p.add_argument("--per-page", type=int, default=8, help="Positions per page for small multiples")
    args = p.parse_args(argv)

    plots = [s for s in (args.plots or "").split(',') if s]
    process_model_dir(args.model_dir, plots=plots, per_page=max(1, int(args.per_page)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
