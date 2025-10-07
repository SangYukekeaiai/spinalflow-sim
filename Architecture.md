


## Layer

### Members

| Member / State                                                |
| ------------------------------------------------------------- |
| `layer_id`                                                    |
| `spine_batches` (mapping: `batch_idx -> [logical_spine_ids]`) |

### Main Functions

| Function                | Parameters Needed      | Parameters Source                                              |
| ----------------------- | ---------------------- | -------------------------------------------------------------- |
| `ComputeSpineBatches()` | `(H, W, mapping_rule)` | Layer-internal configuration (precomputed before Core/ISB use) |

---

## Core

### Members

| Member / State             | Notes                                            |
| -------------------------- | ------------------------------------------------ |
| `input_spine_buf`          | storage: `[kNumPhysISB][kIsbEntries]` of `Entry` |
| `read_index[kNumPhysISB]`  | per-phys buffer read pointer (entries)           |
| `valid_count[kNumPhysISB]` | per-phys buffer valid entry count                |
| `dram_handle`              | SimpleDRAM handle                                |
| `batch_cursor`             | current batch index                              |
| `batches_needed`           | total batches                                    |




### Main Functions

| Function               | Parameters Needed | Parameters Source |
| ---------------------- | ----------------- | ----------------- |
| *(none for this step)* | *(n/a)*           | *(n/a)*           |

---

## InputSpineBuffer (ISB)

### Members

| Member / State                         | Notes                                                    |
| -------------------------------------- | -------------------------------------------------------- |
| `num_phys_spine_buffers = kNumPhysISB` | e.g., 16                                                 |
| `entries_per_buffer = kIsbEntries`     | e.g., 1024                                               |
| `entry_size_bytes = kEntrySizeBytes`   | `sizeof(Entry)`                                          |
| `buffers[kNumPhysISB][kIsbEntries]`    | fixed-capacity `Entry` array                             |
| `buffer_valid_count[kNumPhysISB]`      | valid `Entry` count per phys buffer                      |
| `buffer_read_index[kNumPhysISB]`       | next read position per phys buffer                       |
| `buffer_logical_spine_id[kNumPhysISB]` | current logical spine id per phys buffer (`-1` if empty) |
| `dram_handle`                          | SimpleDRAM handle                                        |



### Main Functions

| Function               | Parameters Needed                                                                             | Parameters Source                                                                                                                                                                         |
| ---------------------- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PreloadFirstBatch()`  | `logical_spine_ids_first_batch`, `layer_id`                                                   | `logical_spine_ids_first_batch`: from Layer → Core; `layer_id`: from Layer → Core                                                                                                         |
| `run()`                | `logical_spine_ids_current_batch`, `layer_id`, `current_batch_cursor`, `total_batches_needed` | `logical_spine_ids_current_batch`: from Core (`spine_batches[current_batch_cursor]`); `layer_id`: from Layer → Core; `current_batch_cursor`: from Core; `total_batches_needed`: from Core |
| `PopSmallestTsEntry()` | *(none)*                                                                                      | *(n/a)*                                                                                                                                                                                 |

## MinFinderBatch
### Members
| Member / State                       | Notes                                                                                             |
| ------------------------------------ | ------------------------------------------------------------------------------------------------- |
| `sf::InputSpineBuffer* isb`          | Non-owning pointer to the input spine buffer                                                      |
| `sf::IntermediateFIFO* fifos`        | Non-owning pointer to a **contiguous array** of IntermediateFIFO (size = `kNumIntermediateFifos`) |
| `sf::Entry picked_entry`             | Internal Entry to hold the selected (popped) item                                                 |
| `bool last_batch_first_entry_pushed` | Flag telling whether the **first entry of the last batch** has been pushed to an IntermediateFIFO |




### Main Functions
| Function                                            | Parameters Needed                        | Parameters Source                                    | How to Implement                                                                                                                                                                                                                                                                                                                        |
| --------------------------------------------------- | ---------------------------------------- | ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run(int current_batch_cursor, int batches_needed)` | `current_batch_cursor`, `batches_needed` | `current_batch_cursor`: Core; `batches_needed`: Core | 1) Select the smallest-ts entry via `isb->PopSmallestTsEntry(picked_entry)`. 2) Validate `current_batch_cursor`, map it to `fifos[current_batch_cursor]`, and `push(picked_entry)`. 3) If `!last_batch_first_entry_pushed && current_batch_cursor == batches_needed - 1 && push succeeded`, set `last_batch_first_entry_pushed = true`. |
| `CanGlobalMegerWork() const`                        | *(none)*                                 | *(n/a)*                                              | Return `last_batch_first_entry_pushed`.                                                                                                                                                                                                                                                                                                 |





## IntermediateFIFO
### Members
| Member / State           | Notes                                                          |
| ------------------------ | -------------------------------------------------------------- |
| `buf_[kCapacityEntries]` | `kCapacityEntries = kInterFifoCapacityBytes / kEntrySizeBytes` |
| `head_`                  | index of oldest entry                                          |
| `size_`                  | current size (entries)                                         |

### Functions
| Function | Parameters Needed | Parameters Source                                                                      |
| -------- | ----------------- | -------------------------------------------------------------------------------------- |
| `run()`  | *(none)*          | Uses `isb` and `fifos` directly: e.g., `isb.PopSmallestTsEntry(...)`, `fifo.push(...)` |


## Global Merger

### Members

| Member / State                                     | Notes                                                                                         |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `IntermediateFIFO (&fifos)[kNumIntermediateFifos]` | **Reference to the array** of IntermediateFIFOs that belong to `MinFinderBatch` (non-owning). |
| `MinFinderBatch& mfb`                              | **Reference** to `MinFinderBatch` (non-owning), used to query `CanGlobalMergerWork()`.        |

### Main Functions

| Function          | Parameters Needed                                                                                      | Parameters Source                                                                         | How to Implement                                                                                                                                                                                                                                                                                                                                                                                                            |
| ----------------- | ------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run(Entry& out)` | 1) A reference to `MinFinderBatch::CanGlobalMergerWork()`; 2) `Entry& out` to receive the popped entry | 1) From `mfb` (call `mfb.CanGlobalMergerWork()`); 2) `out` is the caller-provided storage | **Step 1:** If `!mfb.CanGlobalMergerWork()`, return `false` (cannot work yet). **Step 2:** Iterate over `fifos[0..kNumIntermediateFifos-1]`; for each non-empty FIFO, `front()` to peek the head entry; select the **smallest timestamp** (tie-break by `neuron_id` if desired). **Step 3:** If all FIFOs empty → return `false`. Otherwise `pop()` from the FIFO holding the winner and assign it to `out`; return `true`. |

## PE Array
### Members
| Member / State                         | Notes                                                                    |
| -------------------------------------- | ------------------------------------------------------------------------ |
| `GlobalMerger& gm`                     | **Reference to Global Merger** (non-owning); used to fetch input entries |
| `Entry gm_entry`                       | Entry popped by Global Merger and stored here                            |
| `std::array<uint8_t, 128> weight_row`  | Weight row grabbed from the filter buffer (128 weights)                  |
| `std::array<PE, 128> pe_array`         | The array of 128 PEs                                                     |
| `std::vector<Entry> out_spike_entries` | Output array to store spiked entries for one time                        |


### Main Functions
| Function                                                                               | Parameters Needed                                     | Parameters Source   | How to Implement                                                                                                                                                                                                                                                                                                                        |
| -------------------------------------------------------------------------------------- | ----------------------------------------------------- | ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `InitPEsBeforeLoop(int threshold, int total_tiles, int tile_idx, int h, int w, int W)` | `threshold`, `total_tiles`, `tile_idx`, `h`, `w`, `W` | Core / Layer        | For `pe_idx = 0..127`: `output_id = (total_tiles * 128) * (h * W + w) + (tile_idx * 128) + pe_idx;` then `pe.RegisterOutputId(output_id); pe.SetThreshold(threshold);`                                                                                                                                                                  |
| `GetInputEntryFromGM(const Entry& in)`                                                 | `Entry in`                                            | Global Merger       | `gm_entry = in;` *(kept for flexibility; optional if `run()` calls `gm` directly)*                                                                                                                                                                                                                                                      |
| `GetWeightRow(FilterBuffer& fb)`                                                       | `gm_entry.neuron_id`; `fb`                            | Filter Buffer       | `weight_row = fb.GetRow( fb.ComputeRowId(gm_entry.neuron_id /*, h, w, W if needed */) );`                                                                                                                                                                                                                                               |
| `bool run()`                                                                           | *(none)*                                              | Uses `gm` + members | **Returns `true` if PE array ran this step, else `false`.** Steps: (1) call `gm.run(gm_entry)`; if `false`, return `false`. (2) `GetWeightRow(...)`. (3) For `pe_idx = 0..127`: `pe.Process(gm_entry.ts, weight_row[pe_idx])`; if spiked, push `Entry{gm_entry.ts, pe.output_neuron_id}` to `out_spike_entries`. Finally return `true`. |


## PE
### Members
| Member / State              | Notes                                         |
| --------------------------- | --------------------------------------------- |
| `int32_t vmem`              | Membrane potential                            |
| `int32_t threshold`         | Threshold                                     |
| `uint32_t output_neuron_id` | Output neuron id assigned by PE Array         |
| `bool spiked`               | Whether the PE spiked in the last `Process()` |


### Main Functions
| Function                                      | Parameters Needed      | Parameters Source                                                     | How to Implement                                                                                                                                                                 |
| --------------------------------------------- | ---------------------- | --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `RegisterOutputId(uint32_t outputId)`         | `outputId`             | From PE Array                                                         | `output_neuron_id = outputId;`                                                                                                                                                   |
| `SetThreshold(int32_t th)`                    | `th`                   | From PE Array                                                         | `threshold = th;`                                                                                                                                                                |
| `Process(uint8_t time_steps, int32_t weight)` | `time_steps`, `weight` | `time_steps`: from `gm_entry.ts`; `weight`: from `weight_row[pe_idx]` | `vmem += weight; if (vmem >= threshold) { vmem = 0; spiked = true; } else { spiked = false; }` *(Entry emission is done by PE Array using `output_neuron_id` and `time_steps`.)* |



## Filter Buffer
### Members
| Member / State                                         | Notes                                                                                                           |
| ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------- |
| `rows[kFilterRows][kNumPE]` (each weight is `uint8_t`) | Fixed-capacity storage: `kFilterRows = 4068`, `kNumPE = 128`. One row corresponds to `(input_channel, kh, kw)`. |
| `int C_in`                                             | Total input channels.                                                                                           |
| `int W_in`                                             | Input width (optional for validation/loader math).                                                              |
| `int Kh, Kw`                                           | Kernel height/width.                                                                                            |
| `int Sh, Sw`                                           | Stride (height/width).                                                                                          |
| `int Ph, Pw`                                           | Padding (height/width).                                                                                         |
| `sf::dram::SimpleDRAM* dram`                           | Non-owning pointer/reference to DRAM interface.                                                                 |
| `int h_in_cur, w_in_cur`                               | Per-step inputs updated by `Update()` (kept for schedule/validity checks).                                      |




### Main Functions
| Function                                                                                                        | Parameters Needed                   | Parameters Source | How to Implement                                                                                                                                                                                                                                                                                                                                                                                                            |
| --------------------------------------------------------------------------------------------------------------- | ----------------------------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Configure(int C_in, int W_in, int Kh, int Kw, int Sh, int Sw, int Ph, int Pw, sf::dram::SimpleDRAM* dram_ptr)` | Layer-wise static params + DRAM ptr | Core/Layer        | Assign members; validate `>0` where needed.                                                                                                                                                                                                                                                                                                                                                                                 |
| `Update(int h_out, int w_out)`                                                                                  | Current output coordinates          | Core (per step)   | Set `h_out_cur = h_out; w_out_cur = w_out`.                                                                                                                                                                                                                                                                                                                                                                                 |
| `int ComputeRowId(uint32_t neuron_id) const`                                                                    | `neuron_id`                         | Upstream Entry    | 1) `c_in = neuron_id % C_in`. 2) `pos_in = neuron_id / C_in`. 3) Decode input coords via **members**: `h_in = pos_in / W_in`, `w_in = pos_in % W_in`. 4) Use **members** stride/padding/output-site: `r = h_in - (h_out_cur * S_h - P_h)`, `c = w_in - (w_out_cur * S_w - P_w)`. 5) If `r,c` not in `[0..K_h), [0..K_w)` → return `-1`. 6) Flatten: `row_id = ((c_in * K_h) + r) * K_w + c` (bounds-check `< kFilterRows`). |
| `Row GetRow(int row_id) const`                                                                                  | `row_id`                            | Caller            | Return the row; bounds-check.                                                                                                                                                                                                                                                                                                                                                                                               |
| `uint32_t LoadWeightFromDram(uint32_t layer_id, uint32_t tile_id)`                                              | `layer_id`, `tile_id`               | Core/Layer        | Call `dram->LoadWeightTile(layer_id, tile_id, rows, kFilterRows*kNumPE)` (bytes).                                                                                                                                                                                                                                                                                                                                           |
## Tiled output buffer
### Members
| Member / State                                                | Notes                                                                                       |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| `PEArray& pe_array`                                           | Reference to the PE array (source of `out_spike_entries`).                                  |
| `int tile_id`                                                 | Fixed **tile index provided by Core** (not derived from PEArray).                           |
| `int writeback_cooldown_cycles`                               | Countdown cycles for write-back (stall/latency model).                                      |
| `std::array<std::vector<Entry>, kTilesPerSpine> tile_buffers` | 8 per-tile buffers of `Entry` (size assumed large enough; still check & throw on overflow). |

### Main Functions
| Function                                                   | Parameters Needed            | How to Implement / Behavior                                                                                                                                                                                                            |
| ---------------------------------------------------------- | ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bool run(int tile_id)`                                    | `tile_id` (current tile idx) | If `writeback_cooldown_cycles > 0`, decrement and return `true`. Else if PEArray has outputs, append them into `tile_buffers[tile_id]`, set cooldown to appended count, clear PE outputs, and return `true`. Otherwise return `false`. |
| `bool PeekTileHead(std::size_t tile_id, Entry& out) const` | `tile_id`                    | Return `false` if empty; otherwise copy front entry to `out` and return `true`.                                                                                                                                                        |
| `bool PopTileHead(std::size_t tile_id, Entry& out)`        | `tile_id`                    | Return `false` if empty; otherwise pop front entry into `out` and return `true`.                                                                                                                                                       |
| `void ClearAll()`                                          | *(none)*                     | Clear all per-tile buffers (used after store or when reusing the object across sites if desired).                                                                                                                                      |
| `int writeback_cooldown_cycles() const`                    | *(none)*                     | Accessor for current cooldown value.                                                                                                                                                                                                   |



## Output Sorter
### Members
| Member / State           | Notes                                            |
| ------------------------ | ------------------------------------------------ |
| `TiledOutputBuffer* tob` | Pointer to the Tiled Output Buffer (non-owning). |
| `OutputSpine* out_spine` | Pointer to the Output Spine (non-owning).        |

### Main Functions
| Function      | Parameters Needed | Parameters Source | How to Implement                                                                                                                                                                                                                                                                                                                                                                    |
| ------------- | ----------------- | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bool Sort()` | *(none)*          | Uses members      | **Single-entry pop per call**: iterate the **heads** of `tob->tile_buffers[0..kTilesPerSpine-1]`; among non-empty ones, pick the entry with the **smallest timestamp (ts)** (no tie-break needed per your assumption); pop from that tile buffer; call `out_spine->Push(entry)` (handle capacity errors as throws inside OutputSpine); return `true`. If all empty, return `false`. |


## Output Spine
### Members
| Member / State                                        | Notes                                                        |
| ----------------------------------------------------- | ------------------------------------------------------------ |
| `sf::dram::SimpleDRAM* dram`                          | Pointer to DRAM interface (non-owning).                      |
| `int spine_id`                                        | Spine id set by Core (e.g., `h * W + w`).                    |
| `std::vector<Entry> buf`                              | Local accumulation buffer of output entries.                 |
| `std::size_t capacity_limit = kOutputSpineMaxEntries` | Maximum number of entries allowed in `buf` (from constants). |

### Main Functions
| Function                                             | Parameters Needed | Parameters Source | How to Implement                                                                                                                                                                                                            |
| ---------------------------------------------------- | ----------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bool Push(const Entry& e)`                          | `e`               | Caller            | If `buf.size() >= capacity_limit`, **throw** (capacity exceeded). Else `buf.push_back(e)` and return `true`.                                                                                                                |
| `uint32_t StoreOutputSpineToDRAM(uint32_t layer_id)` | `layer_id`        | Core/Layer        | If `!dram`, **throw**. Compute `bytes = buf.size() * sizeof(Entry)`. Call `dram->StoreOutputSpine(layer_id, spine_id, buf.data(), bytes)`. On success, `buf.clear()` and return the byte count; on DRAM failure, **throw**. |

## Core
### Members
| Member / State                                       | Notes                                                                                                                             |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **External context (provided by conv_layer)**        |                                                                                                                                   |
| `int layer_id`                                       | Current layer id.                                                                                                                 |
| `int h_out, w_out, W_out`                            | Output-spine site coordinates and output width.                                                                                   |
| `const std::vector<std::vector<int>>* spine_batches` | Optional batch table from conv_layer; `(*spine_batches)[k]` is batch `k`.                                                         |
| **Tile control (Core-owned)**                        |                                                                                                                                   |
| `int total_tiles`                                    | Computed by `ConfigureTiles(C_out)` as `C_out / kNumPE` (assumed divisible; padding/error policy is decided by the caller).       |
| **Batching**                                         |                                                                                                                                   |
| `int total_batches_needed`                           | If `spine_batches` present: `spine_batches->size()`; otherwise `ceil(K_h*K_w / kNumPhysISB)` (kernel-slot bound when applicable). |
| `int batch_cursor`                                   | 0-based index of the current batch.                                                                                               |
| **Subsystems / wiring**                              |                                                                                                                                   |
| `sf::dram::SimpleDRAM* dram`                         | DRAM interface.                                                                                                                   |
| `FilterBuffer* fb`                                   | Configured/updated by conv_layer (Configure + `Update(h_out,w_out)` happen **outside** Core).                                     |
| `InputSpineBuffer* isb`                              | Physical ISBs; performs DRAM loads.                                                                                               |
| `IntermediateFIFO fifos[kNumIntermediateFifos]`      | FIFOs between MinFinderBatch and GlobalMerger.                                                                                    |
| `MinFinderBatch mfb`                                 | Uses `isb*`, `fifos*`.                                                                                                            |
| `GlobalMerger gm`                                    | Uses `fifos*`, `mfb&`.                                                                                                            |
| `PEArray pe_array`                                   | Uses `gm`; produces `out_spike_entries`.                                                                                          |
| `TiledOutputBuffer tob`                              | Uses `pe_array&`; holds **all per-tile buffers** (tile = output-spine partition, e.g., every 128 channels).                       |
| `OutputSpine out_spine`                              | Uses `dram*`; `spine_id = h_out*W_out + w_out`.                                                                                   |
| `OutputSorter sorter`                                | Global single-entry merge across **all** tile buffers via `Sort()` per call.                                                      |
| **Per-cycle valid / ran (Core-owned)**               |                                                                                                                                   |
| `bool v_tob_in`                                      | Valid-in for **Stage 0: TiledOutputBuffer ingress**.                                                                              |
| `bool v_pe`                                          | Valid-in for **Stage 1: PEArray.run**.                                                                                            |
| `bool v_mfb`                                         | Valid-in for **Stage 2: MinFinderBatch.run**.                                                                                     |
| `bool ran_tob_in`                                    | Observed result of Stage 0 this cycle.                                                                                            |
| `bool ran_pe`                                        | Observed result of Stage 1 this cycle.                                                                                            |
| `bool ran_mfb`                                       | Observed result of Stage 2 this cycle.                                                                                            |
| **Progress / status (per last `StepOnce` call)**     |                                                                                                                                   |
| `bool compute_finished`                              | Compute loop **for the tile passed to the last `StepOnce(tile_id)`** is done (does **not** include final global drain/store).     |
| `uint64_t cycle`                                     | Optional cycle counter.                                                                                                           |




### Main Functions
| Function                                                        | Parameters Needed                  | How to Implement                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |   |                                                                                                                                                                                                                                                                            |   |        |   |            |
| --------------------------------------------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | - | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | - | ------ | - | ---------- |
| `ConfigureTiles(int C_out)`                                     | `C_out`                            | Compute `total_tiles = C_out / kNumPE` (assume divisible). Typically call `tob.ClearAll()` at the start of each site to reuse buffers. Bind/refresh `sorter` to `(&tob, &out_spine)`. Initialize valids: `v_tob_in = true; v_pe = false; v_mfb = !isb->AllEmpty() && TargetFifoHasSpace();`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |   |                                                                                                                                                                                                                                                                            |   |        |   |            |
| `BindTileBatches(const std::vector<std::vector<int>>* batches)` | optional `batches` from conv_layer | Set `spine_batches = batches; batch_cursor = 0;`. Set/compute `total_batches_needed`. Initialize valids consistent with ISB/FIFO states: `v_tob_in = true; v_pe = false; v_mfb = !isb->AllEmpty() && TargetFifoHasSpace();`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |   |                                                                                                                                                                                                                                                                            |   |        |   |            |
| `PreloadFirstBatch()`                                           | *(none)*                           | Call `isb->PreloadFirstBatch((*spine_batches)[0], layer_id); batch_cursor = 0;`. If `spine_batches` are absent, Core may build a default first batch using the kernel-slot vs `kNumPhysISB` rule.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |   |                                                                                                                                                                                                                                                                            |   |        |   |            |
| `bool StepOnce(int tile_id)`                                    | `tile_id ∈ [0, total_tiles)`       | **Per-cycle compute (no global drain/store here):** <br>**Stage 0 – TOB ingress:** if `v_tob_in` then `ran_tob_in = tob.run(tile_id)`; else `ran_tob_in=false`. <br>**Stage 1 – PEArray compute:** if `v_pe` then `ran_pe = pe_array.run(*fb, h_out, w_out, W_out)`; else `ran_pe=false`. <br>**Stage 2 – MinFinderBatch:** if `v_mfb` then `ran_mfb = mfb.run(batch_cursor, total_batches_needed)`; else `ran_mfb=false`. <br>**Stage 3 – Batch load if needed:** if `isb->AllEmpty()` and `batch_cursor+1 < total_batches_needed`, increment `batch_cursor` and `isb->run((*spine_batches)[batch_cursor], layer_id, batch_cursor, total_batches_needed)`. <br>**Next valids (hard backpressure):** let `cooldown = (tob.writeback_cooldown_cycles() > 0)`, `pe_hasout = !pe_array.out_spike_entries().empty()`, `fifo_has = FifosHaveData()`, `isb_has = !isb->AllEmpty()`, `fifo_space = TargetFifoHasSpace()`. Then `v_tob_in_next = cooldown |   | pe_hasout`; `v_pe_next = (!cooldown) && fifo_has`; `v_mfb_next = (!cooldown) && isb_has && fifo_space`. Commit `v_* = *_next`. <br>**Finish condition (for this `tile_id`):** `compute_finished = (!cooldown) && !fifo_has && !pe_hasout && !isb_has`. Return `(ran_tob_in |   | ran_pe |   | ran_mfb)`. |
| `void DrainAllTilesAndStore()`                                  | *(none)*                           | **Caller must ensure all tiles are fully computed** and `tob.writeback_cooldown_cycles() == 0`. Then repeatedly call `sorter.Sort()` until it returns `false` (all per-tile buffers empty across `tile_id ∈ [0, total_tiles)`). Finally call `out_spine.StoreOutputSpineToDRAM(layer_id)`. Optionally `tob.ClearAll()` to reuse buffers for the next site.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |   |                                                                                                                                                                                                                                                                            |   |        |   |            |
| `bool FifosHaveData() const`                                    | *(none)*                           | Return `true` if any IntermediateFIFO is non-empty.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |   |                                                                                                                                                                                                                                                                            |   |        |   |            |
| `bool TargetFifoHasSpace() const`                               | *(none)*                           | Return `!fifos[batch_cursor].full()` if `batch_cursor` valid; else `false`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |   |                                                                                                                                                                                                                                                                            |   |        |   |            |
| `bool TobEmpty() const`                                         | *(none)*                           | Return `true` if all per-tile buffers inside TOB are empty (helper for site-level draining).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |   |                                                                                                                                                                                                                                                                            |   |        |   |            |
| `bool FinishedCompute() const`                                  | *(none)*                           | Return the `compute_finished` flag corresponding to the **last** `StepOnce(tile_id)` call.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |   |                                                                                                                                                                                                                                                                            |   |        |   |            |


## Conv_layer
### Members
| Member / State                              | Notes                                                                                          |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `int layer_id`                              | Numeric layer identifier.                                                                      |
| `int C_in, C_out`                           | Input/output channels for the layer.                                                           |
| `int H_in, W_in`                            | Input feature-map spatial size.                                                                |
| `int H_out, W_out`                          | Output feature-map spatial size.                                                               |
| `int Kh, Kw`                                | Kernel height/width.                                                                           |
| `int Sh, Sw`                                | Stride height/width.                                                                           |
| `int Ph, Pw`                                | Padding height/width.                                                                          |
| `sf::dram::SimpleDRAM* dram`                | DRAM handle for weight/input/output transfers.                                                 |
| `FilterBuffer fb` *(or pointer/reference)*  | Used for `Configure(...)`, `Update(h_out,w_out)`, and `LoadWeightFromDram(layer_id, tile_id)`. |
| `Core& core`                                | The compute pipeline owner (ISB → MFB → FIFOs → GM → PEArray → TOB → Sorter → OutputSpine).    |
| **Constants (consumed by driver and Core)** | `kNumPhysISB` (e.g., 16), `kNumPE` (=128), `kTilesPerSpine` (upper bound on tiles per spine).  |

### Main Functions
| Parameters Needed                                                 | What it does                                                                                                                                                                                   |
| ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `layer_id, C_in, C_out, H_in, W_in, Kh, Kw, Sh, Sw, Ph, Pw, dram` | Configure the `FilterBuffer` once with layer-wise static parameters and DRAM pointer: `fb.Configure(C_in, W_in, Kh, Kw, Sh, Sw, Ph, Pw, dram)`. Store layer metadata for later site execution. |
