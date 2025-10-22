# SpinalFlow Simulator (CLI)

A simple cache and SRAM behavior simulator with CSV outputs for analysis and plotting.

## Quick Start

- Build the project (example): `mkdir -p build && cd build && cmake .. && make -j && cd ..`
- Run the simulator: `./build/sim <dram_image.bin> <config.json>`

By default:
- Stats CSVs are enabled.
- Reuse-distance CSVs are disabled.
- ts_duration CSVs are disabled (independent of trace).
- Cache trace files are disabled.

## CLI Flags

- `--stats=on|off` or `--no-stats`
  - Master switch for most CSVs (does not force ts_duration if explicitly toggled).
- `--reuse-csv=on|off` or `--no-reuse-csv`
  - Controls per-layer reuse-distance distribution CSVs under `stats/<repo>/<model>/reuse_distance_distribution/`.
- `--scoreboard-csv=on|off` or `--no-scoreboard-csv`
  - Controls scoreboard score CSVs.
- `--setuniq-csv=on|off` or `--no-setuniq-csv`
  - Controls per-set unique demand-line CSVs.
- `--ts-duration-csv=on|off` or `--no-ts-duration-csv`
  - Controls per-layer timestep access CSVs under `stats/<repo>/<model>/ts_duration/`.
  - Independent from `--trace`; you can enable this without enabling cache traces.
- `--trace=on|off` or `--no-trace`
  - Controls detailed cache trace text files.

## Examples

- Enable ts_duration CSVs without traces:
  - `./build/sim dram.bin config.json --ts-duration-csv=on --trace=off`
- Enable reuse-distance CSVs along with stats:
  - `./build/sim dram.bin config.json --stats=on --reuse-csv=on`
- Turn off all CSVs (including ts_duration) quickly:
  - `./build/sim dram.bin config.json --no-stats`

## Plotting Scripts

- Reuse-distance plots: `scripts/reuse_distance_plots.py`
- Timestep duration plots: `scripts/ts_duration_plots.py`
  - Expects CSVs under `stats/<repo>/<model>/ts_duration/`
  - Generate them by running the simulator with `--ts-duration-csv=on`
- Cache trace plots: `scripts/cache_trace_plots.py`
  - Input CSV: `stats/<repo>/<model>/layer<L>/cache_traces/lru/144KB_4ways_0prefetches.csv`
  - Output dir (auto): `.../cache_traces/lru/144KB_4ways_0prefetches/`
  - Per‑position PDFs: `<hout>_<wout>_{counts|rates}.pdf` (clustered bars per timestep; tiles as colors, hit/miss as hatches)
  - Small multiples: `small_multiples_{counts|rates}.pdf` (several positions per page, shared legends)
  - Heatmaps: `heatmaps_{hit_rate|miss_rate}.pdf` (rate per position for each timestep)
  - Overall per‑timestep stacked bars: `overall_per_timestep_{counts|rates}.pdf`
  - CLI examples:
    - `python3 scripts/cache_trace_plots.py` (defaults to the VGG16 L5 path above)
    - `python3 scripts/cache_trace_plots.py --csv build_test_l5/stats/repo4/vgg16/layer5/cache_traces/lru/144KB_4ways_0prefetches.csv`
    - `python3 scripts/cache_trace_plots.py --per-page 8` (small‑multiples density)
    - `python3 scripts/cache_trace_plots.py --plots perpos_counts,overall_rates` (select which plots)
  - `--plots` options (comma‑separated):
    - `all` (default)
    - `perpos`, `perpos_counts`, `perpos_rates`
    - `small`, `small_counts`, `small_rates`
    - `heatmaps`
    - `overall`, `overall_counts`, `overall_rates`

### Batch generation for cache traces

- Sweep a model directory and generate selected figures for every cache_traces CSV:
  - Script: `scripts/cache_trace_sweep.py`
  - Default model dir: `stats/repo4/vgg16`
  - Default plots: `small_counts,overall_counts,heatmaps`
  - Skips generation if a plot already exists for a given CSV.
  - Examples:
    - `python3 scripts/cache_trace_sweep.py` (use defaults)
    - `python3 scripts/cache_trace_sweep.py --model-dir stats/repo4/vgg16 --plots all`
    - `python3 scripts/cache_trace_sweep.py --model-dir stats/repo4/vgg16 --plots small_rates,overall_rates --per-page 10`
