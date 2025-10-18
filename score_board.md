# Eviction Policy — Worked Examples

Below is a concise, **English-only** Markdown version of the eviction policy explanation with several progressively trickier scenarios.

---

## How the policy works (quick recap)

1. **Prefer invalid way first**  
   Scan the set’s ways; if you find `valid == false`, return that index immediately (no scoring/LRU needed).

2. **Otherwise: score → LRU**  
   - For each way, get a **channel score** via `scoreboard_.Get(w.channel_id)`. **Smaller score = colder** (less valuable).  
   - Collect all ways that have the **minimum score** into `candidates`.  
   - Among `candidates`, evict the entry with the **largest `lru_counter`** (i.e., the stalest by your bookkeeping).  
   - If there is a tie on `lru_counter`, the code keeps the **first** candidate (because it only replaces on strictly larger LRU).

---

## 1) Invalid way present beats everything

If any way is `valid == false`, it returns that index immediately—no scoring/LRU needed.

| idx | valid | channel_id | score | lru_counter |
|-----|-------|------------|-------|-------------|
| 0   | true  | 2          | 7     | 5           |
| 1   | **false** | —      | —     | —           |
| 2   | true  | 3          | 1     | 9           |
| 3   | true  | 2          | 1     | 10          |

**Victim:** `1`  
**Reason:** Early return on invalid way.

---

## 2) Unique minimum score wins, even with small LRU

When all ways are valid, pick by **smallest score** first—even if that way’s `lru_counter` is small.

| idx | valid | channel_id | score | lru_counter |
|-----|-------|------------|-------|-------------|
| 0   | true  | 7          | 9     | 1   |
| 1   | true  | 5          | **3** | 4   |
| 2   | true  | 4          | 8     | 200 |
| 3   | true  | 1          | 5     | 300 |

**Victim:** `1`  
**Reason:** Minimum score = 3 is unique at index 1.

---

## 3) Tie on score → break with largest LRU

If several ways share the **same minimal score**, pick the one with **largest `lru_counter`** (stalest).

| idx | valid | channel_id | score | lru_counter |
|-----|-------|------------|-------|-------------|
| 0   | true  | 9          | **2** | 7  |
| 1   | true  | 9          | **2** | 13 |
| 2   | true  | 3          | 5     | 99 |
| 3   | true  | 9          | **2** | 12 |

**Candidates:** `{0, 1, 3}` (score = 2)  
**Victim:** `1` (largest LRU = 13)

---

## 4) All scores equal → pure LRU

If every score is equal, the policy reduces to **evict the largest LRU**.

| idx | valid | channel_id | score | lru_counter |
|-----|-------|------------|-------|-------------|
| 0   | true  | 1          | 4     | 5  |
| 1   | true  | 2          | 4     | 17 |
| 2   | true  | 3          | 4     | 9  |
| 3   | true  | 4          | 4     | 2  |

**Victim:** `1`  
**Reason:** All scores 4 → choose max LRU (17).

---

## 5) Tie on both score **and** LRU → first encountered

If candidates tie on score *and* `lru_counter`, the code keeps the **first** candidate (`best = candidates.front()` and only updates on strictly larger LRU).

| idx | valid | channel_id | score | lru_counter |
|-----|-------|------------|-------|-------------|
| 0   | true  | 6          | **4** | **10** |
| 1   | true  | 7          | **4** | **10** |
| 2   | true  | 8          | 9     | 1      |
| 3   | true  | 9          | **4** | **10** |

**Candidates:** `{0, 1, 3}`  
**Victim:** `0` (the first among equals)





