# Scoreboard-Based Eviction Policy (Clear Guide + Examples)

This document explains how the scoreboard eviction policy works in this simulator, including how time steps are handled and how victims are chosen within a set. It also provides concrete, numeric examples.

---

## Quick Summary

- Set-associative cache; on every miss, choose a victim way within the set.
- Prefer invalid way first. If none, use scoreboard to prefer “colder” channels, then break ties by LRU.
- Time-stepped scoring: the simulator processes time steps `t = 0..T` repeatedly across the whole layer. Scores are aggregated per time step across the entire layer.

---

## Time And Scoreboards

- For each PE/GM entry, we observe a timestamp `t` in `[0..T]`.
- The cache maintains per-timestep cumulative scores across the entire layer:
  - `S[t]`: counts how many spikes/events each input channel produced at time step `t`, across all tiles and output spines in the layer.
  - `S_total = sum_t S[t]`: layer-level cumulative scores for each channel.
- Eviction uses the previous step’s scoreboard:
  - At `t = 0`: use pure LRU (scoreboard not used).
  - At `t ≥ 1`: score for a cache line with `channel_id = c` is `S[t-1][c]`.
- While running at time `t`, the cache increments `S[t]` as spikes arrive. When the next time step occurs, `S[t-1]` is already aggregated over all previous occurrences of `t-1` in the layer.

Why S[t-1]? It avoids looking ahead and makes the decision depend only on statistics that are already stable when operating at time `t`.

---

## Victim Selection In One Set

1) Invalid way first
- If any way has `valid == false`, evict it immediately (no scoring needed).

2) Otherwise: Score → LRU
- Compute `score = S[t-1][w.channel_id]` for each valid way `w` (t=0 uses LRU only, i.e., score treated as 0 for all ways).
- Collect all ways with the minimum score as candidates.
- Among candidates, pick the way with the largest `lru_counter` (stalest).
- If there is still a tie on `lru_counter`, pick the first encountered (stable tie-break).

The simulator’s traces show this value as `score(S[t-1])`.

---

## Concrete Example A: Single Miss At t=1

Assume we are at time `t = 1`, so eviction uses `S[0]`.

- Aggregated per-layer `S[0]` (excerpt):
  - `S[0][2] = 9`, `S[0][9] = 3`, `S[0][4] = 0`, others = 0.
- Set state before fill:

| way | valid | channel_id | lru_counter | scoreboard (`s[0][channel_id]`) |
| --: | :---: | :--------: | ----------: | :-----------------------------: |
|   0 |  true |      2     |           5 |                9                |
|   1 |  true |      9     |          13 |                3                |
|   2 |  true |      4     |          12 |                0                |
|   3 |  true |      7     |           7 |                0                |


- Scores at `t=1` are `S[0][channel]`: 2→9, 9→3, 4→0, 7→0.
- Minimum score = 0 → candidates `{2, 3}`.
- Break tie by LRU: `way 2` has larger `lru_counter` (12 > 7) → evict `way 2`.

---

## Concrete Example B: t=0 Uses LRU Only

At `t = 0`, the policy ignores scoreboard and evicts by LRU only (after preferring invalid way).

| way | valid | channel_id | lru_counter |
|-----|-------|------------|-------------|
| 0   | true  | 3          | 21          |
| 1   | true  | 8          | 10          |
| 2   | true  | 5          | 11          |
| 3   | true  | 6          | 4           |

Minimum invalid? none → choose largest LRU → `way 0` (21).

---

## Concrete Example C: Multiple Passes Across The Layer

Real execution visits each time step many times while sweeping `(h, w)` and tiles:

- As the layer runs, `S[0]`, `S[1]`, …, `S[T]` grow cumulatively. For example, every time the system processes entries at `t=0`, the `S[0]` counts for the touched channels increase.
- When we later evict at `t=1`, we use the current aggregated `S[0]` gathered so far in the entire layer, not just the most recent receptive field. This provides a stable, global notion of “hotness” for the previous time step.

Implication: If your per-step CSV for `t=3` shows small numbers locally, remember that it is the sum over all layer visits to `t=3` (now aggregated). Layer-level totals are in `S_total` and can be much larger.

---

## CSVs And Traces

- Layer scoreboard CSV (cumulative across the layer):
  - `.../layer<L>/scoreboard_scores_<sizeKB>KB_<ways>_<prefetch>_scoreboard.csv`
  - Built from `S_total` (sum over all `t`). Values can be large (hundreds/thousands).

- Per-timestep scoreboard CSVs (aggregated over the layer per step):
  - `.../layer<L>/scoreboard_steps/<policy>/<ways>_<prefetch>/scoreboard_scores_<sizeKB>KB_<ways>_<prefetch>_<policy>_t<t>.csv`
  - Each file contains `S[t]` aggregated over the whole layer.

- Traces (for each miss):
  - `[SET_VALID]` lists each valid way with `score(S[t-1])` and `lru`.
  - `[EVICT]` shows the victim channel and its `score` used in the decision.
  - `[SB][CHOOSE]` records the chosen way, `score(S[t-1])`, `lru`, and dumps the current `S[t-1]` state used.

---

## Tie-Break Worked Examples

1) Invalid way present beats everything

| idx | valid | channel_id | score(S[t-1]) | lru_counter |
|-----|-------|------------|---------------|-------------|
| 0   | true  | 2          | 7             | 5           |
| 1   | false | —          | —             | —           |
| 2   | true  | 3          | 1             | 9           |
| 3   | true  | 2          | 1             | 10          |

Victim: `1` (invalid).

2) Unique minimum score wins, even with small LRU

| idx | valid | channel_id | score(S[t-1]) | lru_counter |
|-----|-------|------------|---------------|-------------|
| 0   | true  | 7          | 9             | 1   |
| 1   | true  | 5          | 3             | 4   |
| 2   | true  | 4          | 8             | 200 |
| 3   | true  | 1          | 5             | 300 |

Victim: `1` (min score = 3).

3) Tie on score → break with largest LRU

| idx | valid | channel_id | score(S[t-1]) | lru_counter |
|-----|-------|------------|---------------|-------------|
| 0   | true  | 9          | 2             | 7  |
| 1   | true  | 9          | 2             | 13 |
| 2   | true  | 3          | 5             | 99 |
| 3   | true  | 9          | 2             | 12 |

Candidates: `{0, 1, 3}` → Victim: `1` (LRU 13 is largest among the min-score set).

4) All scores equal → pure LRU

| idx | valid | channel_id | score(S[t-1]) | lru_counter |
|-----|-------|------------|---------------|-------------|
| 0   | true  | 1          | 4             | 5  |
| 1   | true  | 2          | 4             | 17 |
| 2   | true  | 3          | 4             | 9  |
| 3   | true  | 4          | 4             | 2  |

Victim: `1` (LRU 17).

5) Tie on both score and LRU → first encountered

| idx | valid | channel_id | score(S[t-1]) | lru_counter |
|-----|-------|------------|---------------|-------------|
| 0   | true  | 6          | 4             | 10 |
| 1   | true  | 7          | 4             | 10 |
| 2   | true  | 8          | 9             | 1  |
| 3   | true  | 9          | 4             | 10 |

Candidates: `{0, 1, 3}` → Victim: `0` (first among equals).

---

## Practical Notes

- Large values in the layer scoreboard CSV come from `S_total` (sum over all t). Per-timestep CSVs typically have smaller counts because they distribute activity by time step.
- If you want to experiment with using `S[t]` (same-step) instead of `S[t-1]` for eviction, it’s a one-line change in the victim scoring function; current design intentionally avoids lookahead.
