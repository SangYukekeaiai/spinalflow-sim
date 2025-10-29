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

## Component Overview

- **Core pipeline (`src/core`)** – models the input spine buffer, filter buffer, PE array, FIFOs, and output sorter. It records DRAM load/store time and compute time each layer spends in the pipeline.
- **Layer wrappers (`src/model`)** – drive the core for convolution and fully connected layers, configuring per-layer parameters and collecting statistics.
- **DRAM model (`src/arch/dram`)** – serves packed spine and weight data to the simulator with simple latency modelling.
- **Statistics (`src/utils/stats_io.cpp`)** – minimal utilities that build the timing directory and write `stage_cycles.csv` when requested.
- **Runner (`src/runner/simulation.cpp`)** – orchestrates the layer sequence, wiring DRAM, core, and statistics.

For additional automation you can build custom scripts on top of the `stats/.../timing/stage_cycles.csv` output.
