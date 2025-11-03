# SpinalFlow Simulator (CLI)

SpinalFlow-Sim is a cycle-level model of the SpinalFlow accelerator datapath. It models the DRAM-backed input pipeline, on-chip compute stages, and the collection of output spikes. Cache modelling has been removed so the simulator focuses purely on DRAM traffic and core timing.

## Quick Start

```bash
mkdir -p build
cmake -S . -B build
cmake --build build
./build/bin/spinalflow-sim <dram_image.bin> <config.json> [options]
```

`config.json` describes the layer sequence exported from the tooling flow. `dram_image.bin` holds the packed weights and input spikes expected by the simulator’s DRAM model.

## CLI Options

| Flag | Description |
| --- | --- |
| `--timing-csv`<br>`--timing-csv=on|off|true|false|1|0`<br>`--no-timing-csv` | Opt-in toggle that controls emission of the timing CSV. The file is disabled by default so the simulator produces no CSVs unless this flag is provided. |

Example:

```bash
./build/bin/spinalflow-sim dram.bin config.json --timing-csv
```

## Outputs

When `--timing-csv` is enabled the simulator writes:

- `stats/<repo>/<model>/timing/stage_cycles.csv` — per-layer DRAM load cycles, compute cycles, and store cycles. The repository and model names are inferred from the config path so runs remain organised by workload.

No other CSVs or cache traces are generated.

## Cache Metric Utilities

The repository ships an integration binary that exercises the cache model on the default VGG16 workload and emits per-layer statistics. Build it along with the rest of the project:

```bash
cmake --build build
```

Then run:

```bash
./build/bin/test_cache_vgg16_l6            # cross-layer cache summary
./build/bin/test_cache_vgg16_l6 --reuse-dist      # also capture reuse-distance histograms
./build/bin/test_cache_vgg16_l6 --spike-stats     # dump spiking event stats per tile
./build/bin/test_cache_vgg16_l6 --reuse-in-tile   # emit per-tile reuse distributions
```

The binary reads the workload assets from `workloads/repo4/vgg16/` and produces:

- `stats/repo4/vgg16/cache_stats.csv` – running log of cache timing and hit ratios.
- `stats/repo4/vgg16/cross_layer_comparsion_stats/288KB_32ways_lru.csv` – per-layer demand and latency metrics in the requested format.
- `stats/repo4/vgg16/reuse_distance_distribution/layer<ID>.csv` – (enabled via `--reuse-dist`) reuse-distance histograms for each layer, plus a boxplot under `stats/repo4/vgg16/reuse_distance_distribution/box/`.
- `stats/repo4/vgg16/spiking_event_stats/layer_<ID>.csv` – (enabled via `--spike-stats`) per-output-spine, per-tile spike counts for each time step.
- `stats/repo4/vgg16/reuse_in_tiles_statistics/layer<ID>.csv` – (enabled via `--reuse-in-tile`) per-output-spine histograms of intra-tile reuse counts.

To mirror the archived plots you can regenerate the box figure with the helper script:

```bash
python3 scripts/plot_reuse_distribution.py \
  --input-dir stats/repo4/vgg16/reuse_distance_distribution \
  --xmin 0 --qright 0.95

python3 scripts/plot_spike_event_distribution.py \
  --input-dir stats/repo4/vgg16/spiking_event_stats \
  --ymax 2000

python3 scripts/plot_reuse_in_tile_distribution.py \
  --input-dir stats/repo4/vgg16/reuse_in_tiles_statistics
```

Each helper accepts clipping arguments (`--xmin/--xmax/--qleft/--qright` for reuse distance; `--ymin/--ymax/--qleft/--qright` for spike and intra-tile reuse counts) and supports an optional `--output` flag. When omitted, PDFs are written alongside the source CSVs (to `<input-dir>/box/` when applicable).

## Component Overview

- **Core pipeline (`src/core`)** – models the input spine buffer, filter buffer, PE array, FIFOs, and output sorter. It records DRAM load/store time and compute time each layer spends in the pipeline.
- **Layer wrappers (`src/model`)** – drive the core for convolution and fully connected layers, configuring per-layer parameters and collecting statistics.
- **DRAM model (`src/arch/dram`)** – serves packed spine and weight data to the simulator with simple latency modelling.
- **Statistics (`src/utils/stats_io.cpp`)** – minimal utilities that build the timing directory and write `stage_cycles.csv` when requested.
- **Runner (`src/runner/simulation.cpp`)** – orchestrates the layer sequence, wiring DRAM, core, and statistics.

For additional automation you can build custom scripts on top of the `stats/.../timing/stage_cycles.csv` output.
