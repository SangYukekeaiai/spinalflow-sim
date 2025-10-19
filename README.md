SpinalFlow Simulator — Notes and Usage

Overview
- This repository contains a C++ simulator for the SpinalFlow SNN accelerator.
- Outputs stats and traces under the `stats/` directory for downstream analysis and plotting.

Build
- Requires CMake 3.16+ and a C++17 compiler.
- Typical flow:
  - `mkdir -p build && cd build`
  - `cmake ..`
  - `cmake --build .` (or `make`)

Key Binaries
- `spinalflow-sim`: main simulator (drives full network via `src/main.cpp`).
- Test/sweep tools under `tests/` (enabled if sources exist):
  - `test_first_layer_cache`
  - `test_layer5_cache`
  - `test_layer8_cache`

Output Layout
- Base: `stats/<repo>/<model>`
- Per-configuration CSV (model-level):
  - `<sizeKB>KB_<ways>_<prefetch>_<policy>.csv`
- Per-layer directory: `layer<L>/`
  - Per-configuration CSV for that layer: same filename as model-level CSV
  - Optional reuse distribution CSV: `reuse_distribution_<sizeKB>KB_<ways>_<prefetch>_<policy>.csv`
  - Optional per-set unique demand lines CSV: `set_unique_demand_lines_<sizeKB>KB_<ways>_<prefetch>_<policy>.csv`
  - Optional cache traces: `cache_traces/<policy>/<ways>_<prefetch>/<sizeKB>.txt`
- Aggregated totals (model-level):
  - `cache_totals_<ways>ways_<prefetch>prefetchs_<policy>.csv`
- Aggregated totals (per-layer):
  - `layer<L>/cache_totals_<ways>ways_<prefetch>prefetchs_<policy>.csv`

Single-Layer Runs
- When sweeping a single layer (i.e., config contains exactly one layer), the simulator writes stats only under `layer<L>/` for that layer:
  - No model-level per-configuration CSV is emitted.
  - No model-level aggregated totals CSV is emitted.
  - Cache traces are also placed under `layer<L>/`.
  - Reuse distributions are per-layer only.

Cache Trace Files
- The simulator can emit detailed cache traces when enabled in code.
- When running a single-layer sweep (i.e., only one layer in `specs`), trace files are placed under that layer’s folder:
  - `stats/<repo>/<model>/layer<L>/cache_traces/<policy>/<ways>_<prefetch>/<sizeKB>.txt`
- For multi-layer runs, traces remain at the model level to avoid ambiguity:
  - `stats/<repo>/<model>/cache_traces/<policy>/<ways>_<prefetch>/<sizeKB>.txt`

Controlling CSV Generation (CLI)
- The main simulator and the test binaries accept command‑line flags to control output:
  - `--stats=on|off|true|false|1|0` or `--no-stats`
    - Enables/disables all stats CSVs (per‑config, per‑layer, totals).
  - `--reuse-csv=on|off|true|false|1|0` or `--no-reuse-csv`
    - Enables/disables only the reuse‑distribution CSVs.
  - `--setuniq-csv=on|off|true|false|1|0` or `--no-setuniq-csv`
    - Enables/disables only the per‑set unique‑address CSVs.
  - `--trace=on|off|true|false|1|0` or `--no-trace`
    - Enables/disables detailed cache trace files.
  - If `--stats` is provided and a specific CSV flag is not, that flag follows the `--stats` setting.

Examples
- Full model with stats off (no CSVs):
  - `bin/spinalflow-sim <dram_image.bin> <config.json> --stats=off`
- Full model with only reuse distributions off:
  - `bin/spinalflow-sim <dram_image.bin> <config.json> --reuse-csv=off`
- Full model with only per‑set unique CSVs off:
  - `bin/spinalflow-sim <dram_image.bin> <config.json> --setuniq-csv=off`
- Full model with traces disabled:
  - `bin/spinalflow-sim <dram_image.bin> <config.json> --trace=off`
- L=5 sweep with stats off:
  - `bin/test_layer5_cache <dram_image.bin> <config.json> --stats=off`
- L=8 sweep with only reuse distributions off:
  - `bin/test_layer8_cache <dram_image.bin> <config.json> --reuse-csv=off`

Utilities Refactor
- Stats helpers have been factored into a clear `utils/` module:
  - Headers: `include/utils/stats_io.hpp`, `include/utils/stats_types.hpp`
  - Source: `src/utils/stats_io.cpp`
- `src/runner/simulation.cpp` now focuses on orchestration and uses these helpers.

Notes
- Defaults are stats on and reuse CSV on if no flags are provided.
- Paths in this README are relative to the repo root or the simulator’s working directory.
