# Group-closure experiment — complete results

**Date:** 11 September 2026 · supersedes `group_closure_partial.md`
**Config:** 20 000 steps, batch 128, lr 1e-3 constant, 1 layer, 12 heads × 32,
seed 666, k = 128, 100 000 train / 2 000 test. Single seed.

**The question:** is the computation a symbol requires a property of that symbol,
or of the whole alphabet?

---

## Full table (token accuracy)

| arm | tokens | ⟨A⟩ | D_max | n_h=2 | n_h=3 | n_h=4 |
|---|---|---|---|---|---|---|
| `A5_c5` | 24 | A₅ | 2 | 0.1259 | **0.9991** | 0.9983 |
| `A5_c5_dt` | 39 | A₅ | 2 | 0.0443 | **0.9994** | 0.0429 |
| `A5_c5_3c` | 44 | A₅ | 2 | **0.9510** | **0.9952** | 0.9993 |
| `S5_c5_t1` | 25 | S₅ | 4 | 0.0904 | **0.1094** | 0.4851 |
| `S5_c5_t` | 34 | S₅ | 4 | 0.0422 | **0.0307** | 0.0706 |

---

## 1. The prediction held, in its sharpest form

```
A5_c5      24 five-cycles                    n_h=3  ->  0.9991
S5_c5_t1   the same 24 + ONE transposition   n_h=3  ->  0.1094
```

One extra symbol. The cheapest symbol that exists (ℓ = 1). **89 accuracy points
destroyed.**

The mixture barely moved: `A5_c5` is 100% five-cycles, `S5_c5_t1` is 96%
five-cycles. What changed is that the alphabet's closure went from A₅ (60
elements, representable in 3 dimensions) to S₅ (120, needing 4).

## 2. The separation at n_h = 3 is total

| | n_h = 3 results |
|---|---|
| A₅ alphabets | 0.9991, 0.9994, 0.9952 |
| S₅ alphabets | 0.1094, 0.0307 |

Worst A₅ result is **0.9952**. Best S₅ result is **0.1094**. A gap of 0.885 with
no overlap, across five independently generated datasets. The predicted variable
— which group the alphabet generates — sorts them perfectly.

## 3. Alphabet size is decisively ruled out

This was the control, and it fired hard:

| | tokens | n_h=2 | n_h=3 |
|---|---|---|---|
| `A5_c5_3c` | **44** | **0.9510** | 0.9952 |
| `S5_c5_t1` | **25** | 0.0904 | 0.1094 |

The alphabet with **nearly twice as many symbols** is the easiest arm in the
entire experiment. The smaller one fails at both orders. Difficulty does not
track vocabulary size; it tracks group closure.

## 4. D = 2 for A₅ is now confirmed empirically, not just proved

`A5_c5_3c` at **n_h = 2** reached **0.9510** (loss 4.87 → 0.17). This is the
first time a two-factor model has worked on any A₅ task.

It matters because ℓ says a five-cycle costs 4. If ℓ were the demand, a
two-factor model could not get near this task at any accuracy. It did. The
demand really is 2, and the model really does find the 3-dimensional
representation on its own.

(It did not cross the 0.99 threshold — loss plateaued at 0.17 — so the script
scored it as a learnability gap. Treat it as "capacity confirmed, convergence
incomplete", not as a failure.)

## 5. Zero refuting cells

Nothing solved below its bound. The one outcome that could have killed the
demand model did not occur in any of the 15 cells.

---

## What this does NOT establish — the honest gap

**n_h = 4 did not solve either S₅ arm.** 0.4851 and 0.0706 against a predicted
solve.

So we have shown that n_h=3 fails on S₅ alphabets, but **not** that n_h=4
succeeds on them — which means we have not demonstrated that the threshold sits
exactly at 4. A skeptic can say: "the S₅ arms are simply hard to optimise, and
n_h=3's failure is an optimisation failure, not a capacity one." This run alone
cannot refute that.

The loss traces say it is under-training rather than incapacity:

```
S5_c5_t1  n_h=4:  4.8697 -> 2.1727 (5k) -> 2.1602 (10k) -> 2.1221 (15k) -> 1.8296 (20k)
S5_c5_t1  n_h=3:  4.8690 -> 3.8781      -> 3.7790       -> 3.7486       -> 3.6543
```

n_h=4 was **still descending** at step 20 000 and sat 1.8 nats below n_h=3.
It was on its way and ran out of budget. But "was on its way" is not a result.

**Why the S₅ arms are hard to train at all** is consistent with the mixture-rate
cliff already recorded in PLAN.md §6.2: `S5_c5_t` samples uniformly over 34
symbols, so 71% of its tokens are five-cycles — squarely in the difficult zone
where the p=0.5 sweep run also stalled. `S5_c5_t1` is 96% five-cycles and did
better (0.4851 vs 0.0706), consistent with that same curve.

### The one run that closes this

**`S5_c5_t1` at n_h = 4, 80 000 steps.** If it solves, the threshold is
demonstrated end to end: fails at 2, fails at 3, solves at 4, on an alphabet one
token away from one that solves at 3. That single cell converts the strongest
result in the project from "strongly suggestive" to "shown".

```bash
python work/exp_group_closure.py --arms S5_c5_t1 --orders 4 --steps 80000 \
    --out results/group_closure_s5t1_long.json
```

≈50 min. Run this before anything else.

### After that

Seeds on the headline pair — `A5_c5` n_h=3 and `S5_c5_t1` n_h=3, five seeds
each. Outcomes on this task are bimodal (PLAN.md §6.2 / partial-results
Reading 3), so the right statistic is the fraction of seeds that escape the
plateau, not a mean.

---

## Claim status after this run

| claim | before | after |
|---|---|---|
| **2 — demand depends on the alphabet** | proved algebraically, untested empirically | **empirically supported, 5 alphabets, perfect separation at n_h=3; threshold not yet pinned at 4** |
| **3 — representable ≠ learnable** | one data point | **four more**: `A5_c5` n_h=2, `A5_c5_dt` n_h=2 and n_h=4, `S5_c5_t1` n_h=4 |
| **1 — capability at matched compute** | 3 of 5 sweep points | unchanged; still single-seed, still needs re-running |

The `A5_c5_dt` n_h=4 failure (0.0429) next to its own n_h=3 success (0.9994)
remains the single most striking optimisation result we have: strictly more
capacity, total failure, same seed and data.
