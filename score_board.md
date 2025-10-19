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


1: 5, 20, 45, 47, 53, 92, 134, 151, 163, 170, 184, 203, 217, 251, 254
2: 17, 22, 50, 54, 77, 104, 140, 158, 160, 187, 237, 239
3: 7, 8, 11, 28, 34, 35, 80, 86, 102, 105, 106, 107, 121, 128, 147, 148, 149, 150, 152, 156, 161, 164, 173. 


(cin=5, score=1), (cin=20, score=1), (cin=45, score=1), (cin=47, score=1), (cin=53, score=1), (cin=92, score=1), (cin=134, score=1), (cin=151, score=1), (cin=163, score=1), (cin=170, score=1), (cin=184, score=1), (cin=203, score=1), (cin=217, score=1), (cin=251, score=1), (cin=254, score=1), (cin=17, score=2), (cin=22, score=2), (cin=50, score=2), (cin=54, score=2), (cin=77, score=2), (cin=104, score=2), (cin=140, score=2), (cin=158, score=2), (cin=160, score=2), (cin=187, score=2), (cin=237, score=2), (cin=239, score=2), (cin=7, score=3), (cin=8, score=3), (cin=11, score=3), (cin=28, score=3), (cin=34, score=3), (cin=35, score=3), (cin=80, score=3), (cin=86, score=3), (cin=102, score=3), (cin=105, score=3), (cin=106, score=3), (cin=107, score=3), (cin=121, score=3), (cin=128, score=3), (cin=147, score=3), (cin=148, score=3), (cin=149, score=3), (cin=150, score=3), (cin=152, score=3), (cin=156, score=3), (cin=161, score=3), (cin=164, score=3), (cin=178, score=3), (cin=183, score=3), (cin=195, score=3), (cin=220, score=3), (cin=243, score=3), (cin=246, score=3), (cin=250, score=3), (cin=135, score=4), (cin=157, score=4), (cin=166, score=4), (cin=168, score=4), (cin=196, score=4), (cin=27, score=5), (cin=78, score=5), (cin=120, score=5), (cin=159, score=5), (cin=222, score=5), (cin=227, score=5), (cin=4, score=6), (cin=56, score=6), (cin=62, score=6), (cin=98, score=6), (cin=108, score=6), (cin=117, score=6), (cin=127, score=6), (cin=176, score=6), (cin=186, score=6), (cin=189, score=6), (cin=205, score=6), (cin=211, score=6), (cin=219, score=6), (cin=230, score=6), (cin=33, score=7), (cin=67, score=8), (cin=71, score=8), (cin=113, score=8), (cin=132, score=8), (cin=136, score=8), (cin=182, score=8), (cin=241, score=8), (cin=252, score=8), (cin=21, score=9), (cin=68, score=9), (cin=103, score=9), (cin=130, score=9), (cin=225, score=9), (cin=226, score=9), (cin=49, score=10), (cin=52, score=10), (cin=60, score=10), (cin=118, score=10), (cin=65, score=11), (cin=175, score=11), (cin=181, score=11), (cin=197, score=11), (cin=26, score=12), (cin=32, score=12), (cin=218, score=12), (cin=244, score=12), (cin=1, score=13), (cin=29, score=13), (cin=63, score=13), (cin=70, score=13), (cin=119, score=13), (cin=169, score=13), (cin=173, score=13), (cin=174, score=13), (cin=39, score=14), (cin=69, score=14), (cin=88, score=14), (cin=89, score=14), (cin=202, score=14), (cin=221, score=14), (cin=224, score=14), (cin=234, score=14), (cin=141, score=15), (cin=236, score=15), (cin=10, score=16), (cin=15, score=16), (cin=51, score=17), (cin=206, score=17), (cin=212, score=17), (cin=55, score=18), (cin=83, score=18), (cin=207, score=18), (cin=133, score=19), (cin=144, score=19), (cin=162, score=19), (cin=109, score=21), (cin=114, score=21), (cin=126, score=21), (cin=165, score=21), (cin=19, score=22), (cin=6, score=23), (cin=59, score=23), (cin=200, score=23), (cin=171, score=24), (cin=185, score=24), (cin=242, score=24), (cin=40, score=25), (cin=100, score=25), (cin=154, score=25), (cin=191, score=25), (cin=235, score=25), (cin=112, score=26), (cin=146, score=26), (cin=38, score=27), (cin=194, score=27), (cin=238, score=27), (cin=46, score=28), (cin=61, score=28), (cin=193, score=28), (cin=228, score=28), (cin=247, score=28), (cin=74, score=30), (cin=115, score=30), (cin=13, score=31), (cin=84, score=31), (cin=248, score=31), (cin=249, score=31), (cin=143, score=32), (cin=177, score=32), (cin=3, score=33), (cin=23, score=34), (cin=43, score=34), (cin=81, score=35), (cin=125, score=37), (cin=137, score=37), (cin=192, score=37), (cin=42, score=38), (cin=240, score=39), (cin=94, score=40), (cin=255, score=40), (cin=210, score=41), (cin=18, score=45), (cin=139, score=45), (cin=87, score=48), (cin=91, score=48), (cin=85, score=51)