# Group-closure experiment — partial results (run interrupted)

**Date:** 11 September 2026
**Config:** 20 000 steps, batch 128, lr 1e-3 constant, 1 layer, 12 heads × 32,
seed 666, k = 128, 100 000 train / 2 000 test
**Status:** arms `A5_c5` and `A5_c5_dt` complete. Stopped during `A5_c5_3c`.
Remaining: `A5_c5_3c`, `S5_c5_t1`, `S5_c5_t`.

---

## Results so far

| arm | tokens | ⟨A⟩ | D_max | n_h | token acc | seq acc | time | vs prediction |
|---|---|---|---|---|---|---|---|---|
| `A5_c5` | 24 | A₅ | 2 | 2 | 0.1259 | 0.0000 | 540s | predicted solve — **failed** |
| `A5_c5` | 24 | A₅ | 2 | 3 | **0.9991** | 0.9360 | 601s | predicted solve — ✅ |
| `A5_c5` | 24 | A₅ | 2 | 4 | **0.9983** | 0.8940 | 760s | predicted solve — ✅ |
| `A5_c5_dt` | 39 | A₅ | 2 | 2 | 0.0443 | 0.0000 | 443s | predicted solve — **failed** |
| `A5_c5_dt` | 39 | A₅ | 2 | 3 | **0.9994** | 0.9530 | 592s | predicted solve — ✅ |
| `A5_c5_dt` | 39 | A₅ | 2 | 4 | **0.0429** | 0.0000 | 754s | predicted solve — **failed** |

---

## Reading 1 — the demand model is not yet contradicted

The capacity bound is a **lower** bound: `n_h ≥ D` is necessary, not sufficient.
That makes the four possible outcomes asymmetric, and only one of them can
falsify the model:

| predicted | observed | meaning |
|---|---|---|
| fail (n_h < D) | fail | consistent |
| solve (n_h ≥ D) | solve | consistent |
| solve (n_h ≥ D) | **fail** | learnability gap — **does not refute the model** |
| **fail (n_h < D)** | **solve** | **refutes the model** |

**We have zero instances of the refuting case.** Every mismatch so far is
"could have, didn't" — an optimisation failure, not a capacity one.

This also means the decisive cells are still ahead of us: `S5_c5_t1` and
`S5_c5_t` at n_h = 2 and 3. Those are predicted to fail. If either **solves**,
the demand model is wrong and Claim 2 dies. The A₅ arms can only ever be
corroborating.

---

## Reading 2 — the important result: more capacity made it fail

```
A5_c5_dt   n_h=3   token 0.9994   <- solves
A5_c5_dt   n_h=4   token 0.0429   <- complete failure
```

Same data, same seed, same everything, **more capacity**. An n_h=4 model can do
everything an n_h=3 model can (set the extra β to 0 — proved constructively in
`exp_beta_reachability.py`). So this is purely an optimisation failure, and it
is the strongest evidence yet for Claim 3.

The loss traces show what happened:

```
n_h=3:  4.8660 → 3.9928 (5k) → 0.0102 (10k) → 0.0038   escaped the plateau
n_h=4:  4.8699 → 3.9963 (5k) → 3.9780 (10k) → 3.9652   never escaped
```

Both sat on a plateau near **3.96–4.00**. One escaped between step 5 000 and
10 000; the other never did within 20 000 steps. Chance level for a 60-target
task is ln(60) = 4.094, so the plateau is a hair below chance — the model has
learned the marginal distribution and nothing else.

The same plateau appeared in the matched-compute sweep at p = 0.5, where
`fixed4` sat at 4.67 until step 11 000 and the oracle sat at ~3.94 until step
16 000 before dropping to 0.07. Three independent sightings now.

---

## Reading 3 — this changes how we must report every number

Outcomes on this task are **bimodal, not noisy**. A run either escapes the
plateau and lands near 0.999, or it does not and lands near 0.04. There is no
middle. So:

- Reporting a mean over seeds is meaningless.
- The right statistic is **the fraction of seeds that escape**, plus the
  accuracy conditional on escaping.
- Single-run comparisons between two configurations are close to worthless
  unless both escaped.

This is a bigger problem than the bf16 noise floor recorded in PLAN.md §5.3
(±0.005 token / ±0.05 sequence). That floor applies *within* the escaped mode.
The bimodality dominates it.

**Consequence for work already done:** the matched-compute sweep is single-seed.
Cells where a fixed arm scored near chance may be plateau failures rather than
capacity failures. The three CLAIM HOLDS verdicts (p = 0.05 / 0.1 / 0.25) rest
on the oracle solving and `fixed1`/`fixed2` failing — and `fixed1`/`fixed2`
failing could in principle be plateau failures too. The claim is not overturned,
because `fixed3`'s partial success at p = 0.1 shows a genuine capacity gradient,
but **the sweep needs re-running with ≥5 seeds per cell before it goes in a
paper.**

---

## Reading 4 — minor

`A5_c5` at n_h=3 (seq 0.9360) vs n_h=4 (seq 0.8940): a 0.042 difference, at the
noise floor. Not meaningful. Do not read anything into n_h=3 "beating" n_h=4
there.

---

## To resume

```bash
python work/exp_group_closure.py \
    --arms A5_c5_3c,S5_c5_t1,S5_c5_t \
    --out results/group_closure_rest.json
```

The `A5_c5` and `A5_c5_dt` numbers above are recorded and do not need re-running
unless we are adding seeds.

**The two cells that decide Claim 2 are `S5_c5_t1` at n_h=2 and n_h=3.** If both
fail, the group-closure account survives its sharpest test. If either solves, it
is wrong.

Given Reading 3, a plateau failure and a capacity failure look identical from
the outside. So when `S5_c5_t1` at n_h=3 fails — as predicted — that on its own
is **weak** evidence, because it could be a plateau failure. The strong evidence
is the pairing: `A5_c5` at n_h=3 solving (0.9991) while `S5_c5_t1` at n_h=3
fails, on alphabets that differ by exactly one token. That contrast is what
needs seeds, and it is worth running 5 of each on just those two cells before
anything else.
