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

