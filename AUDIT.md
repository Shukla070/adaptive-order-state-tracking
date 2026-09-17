# Methodology audit — 13 September 2026

Companion to `PLAN.md`. Written after the β-parametrisation experiment, which
found three bugs in our own tooling and one measurement-design error before it
found anything about the architecture.

**Purpose:** classify every claim by how badly the errors touch it, so that the
next document we write does not inherit an unestablished number. Nothing here
supersedes `PLAN.md`; the corrections it implies are listed in §6 and should be
applied to `PLAN.md` once the escape census (§5) returns.

**Short answer to "was it all our own workflow?"** The errors were ours, and
none of them touch the proved algebra. But the consequence is not small: they
mean several *measured* claims — including the central high-demand training wall
— are currently unestablished rather than merely imprecise.

---

## 1. The errors

| # | What | Where | Effect | Status |
|---|---|---|---|---|
| 1 | Accuracy on an empty set reported as 0.0 | `exp_beta_param.py`, `acc_odd = od_c / max(1, od_n)` | Printed a "parity wall, showing up in a trained network" verdict on every A₅ cell. A₅ *is* the even half, so those cells have no odd targets — the split was undefined, not failed. Fired on all three arms including the one that scored 0.9997. | fixed; `parity_report` returns `None`, `--selftest` T2/T3 cover it |
| 2 | Evaluation truncated to 384 sequences | same file, `if i >= 256: break` in the eval loop | Every headline token/sequence accuracy came from 384 sequences while every other script in the project reports over 2000. Not comparable, and noisier. | fixed; accuracy runs the full test set, only the β probe is capped |
| 3 | Verdict compared against a hardcoded number from a different script | same file, `BASELINE` dict | Printed **FIXED — 0.1259 → 0.9997** when the unmodified `sigmoid(0)` control *in the same run* had scored 0.9598. | fixed; `HISTORICAL` is display-only, the verdict uses the in-run control, and an attribution warning fires when the two disagree |
| 4 | Single-seed reporting on bistable cells | project-wide | See §3. The largest of the four. | being measured (§5) |
| 5 | Verdict compared medians of a bimodal distribution | the fix for #3 | With 2/4 escapes the median jumps to the upper mode, so 2/4 vs 1/4 produced medians of 0.9791 vs 0.1237 — an apparent 8× effect from a difference with Fisher p = 1.000. | fixed; verdict tests escape fractions with an exact test |
| 6 | No self-test, contrary to the working agreement | `exp_beta_param.py` | Errors 1 and 2 would both have been caught by one. | fixed; 7 known-answer tests, verified to FAIL against the old implementation |

Errors 1, 2, 3 and 6 are tooling. Error 4 is experiment design, and it is the
one that changes conclusions.

---

## 2. What is not affected

**The proved algebra is untouched.** It is computed in float64 on CPU by
explicit construction, with no optimiser, no bf16 kernel, and no sampling over
runs. None of the failure modes above can reach it.

- `K ≥ ℓ` forced by rank — `exp_beta_reachability.py` E1
- the parity wall, β pinned ⟹ `K ≡ ℓ (mod 2)` — all 120 elements, worst error 1.17e-15
- no wider representation of S₅ helps — E4
- `D(g) = min over faithful ρ of rank(ρ(g) − I)`; a 5-cycle costs 4 in S₅ and 2
  in A₅ — `exp_minimal_order.py` V1–V5, icosahedral group built from geometry,
  two-reflection factorisation to 1.31e-15

This is the strongest part of the project and it survives intact. It is also
the part that carries the honest value proposition: `n_h` is a hyperparameter
nobody can choose correctly, because the required value depends on the closure
of the whole alphabet. That claim needs no training run to be true.

**The two reproductions are safe.** S₃ matches the published 1/0 pattern; S₅ at
`n_h=4` reached token 0.864 at length 512 trained at 128. Both sit deep inside
a solved basin, far from the knife edge described below.

---

## 3. The measurement error, and what it costs

### The observation

`A5_c5:n_h=2`, identical configuration, identical seed (666), identical data
seed, byte-identical step-1 loss of 4.8687:

| | step 1 | step 5000 | step 20000 | token acc |
|---|---|---|---|---|
| run of 13 Sept, 09:00 | 4.8687 | **0.5250** | 0.1578 | **0.9598** |
| run of 13 Sept, 09:34 | 4.8687 | **3.7125** | 3.6140 | **0.1177** |

The only uncontrolled variable is bf16 kernel non-determinism. By step 5000 it
has already decided which basin the run lands in.

**So the seed does not control the outcome on this cell.** Seed is not the unit
of replication, and re-running with a fixed seed is not a reproduction.

Pooled observations so far on `A5_c5:n_h=2` (n=4 per arm, provisional):

```
sigmoid(0)   0.1177  0.1204  0.1237  0.9598     escapes 1/4
clamp(2)     0.1160  0.1166  0.9791  0.9997     escapes 2/4
                                                Fisher exact p = 1.000
```

### What this costs us

**The noise floor in `PLAN.md` §5.3 is wrong as a general figure.** The quoted
≈0.005 token / 0.05 sequence was measured on `fixed3` at p=0.1 — a cell that
reliably solves. On a bistable cell the spread at fixed seed is **0.84 token
accuracy**. Applying the stable-cell figure to bistable cells is what licenses
single-seed claims.

**Every mid-range number in a single-seed table is suspect.** Values near
0.999 or near 0.04 are probably deep in a basin and robust. Values in between —
`fixed2` 0.4659, `fixed3` 0.5516, `fixed4` 0.7070, `fixed3` 0.8689 — are
exactly where a bistable cell lands when sampled once, and cannot be read as
capability measurements.

**The high-demand training wall is weakly supported.** `PLAN.md` Part 3b reports
0 of 4 high-demand cells training versus 6 of 7 low-demand. If each cell is a
coin flip with escape probability `q`:

| q | P(0 of 4 escape) | P(≥6 of 7 escape) |
|---|---|---|
| 0.10 | 0.656 | 0.00001 |
| 0.25 | **0.316** | 0.00134 |
| 0.33 | 0.202 | 0.00648 |
| 0.50 | 0.062 | 0.06250 |

If the high-demand cells escape as often as `A5_c5:2` provisionally appears to,
**"0 of 4" occurs 32% of the time by chance.** That is not evidence of a wall.

The *contrast* survives: "6 of 7" is essentially impossible under any bistable
model, so the low-demand cells genuinely are reliably trainable. What does not
survive is the strong form — that high-demand cells *cannot* train. They may
simply train rarely, which is a different claim with different consequences.

Note this cuts both ways and one direction is good news: a cell that escapes
25% of the time is not an architecture that cannot learn the task. It is an
optimisation problem with a bad basin structure, and those are more tractable
than representational limits.

---

## 4. Derived numbers that still have no test

Errors 1 and 5 were both derived statistics that looked plausible and were
wrong. Two others of the same class are load-bearing and untested:

- **The 0.545 A₅ ceiling** (`PLAN.md` Part 3b), used to interpret the 250k run
  as "the models learn the even structure and stall on the parity bit." It is
  described as computed rather than fitted, but it has no self-test, and the
  interpretation of our longest run rests on it.
- **`effective_cost` / `predicted_demand`** — the functions producing the
  oracle's per-token cost figures in Claim 1.

Neither is known to be wrong. Both should get a known-answer test before they
appear in another document.

---

## 5. RESULT: the escape census (was pending; measured 13 Sept)

**Every cell is bistable. Not one is reliably broken.** 33 runs completed of 40
planned; the run crashed at #34 on an out-of-vocabulary parity index (error #7
below).

| cell | historical (n=1) | escapes / runs | rate | 95% CI |
|---|---|---|---|---|
| `A5_c5:2` | 0.1259 | 3 / 10 | 0.30 | [0.11, 0.60] |
| `A5_c5_dt:2` | 0.0443 | 2 / 10 | 0.20 | [0.06, 0.51] |
| `A5_c5_dt:4` | 0.0429 | 1 / 10 | 0.10 | [0.02, 0.40] |
| `S5_c5_t:4` | 0.0706 | 1 / 3 | 0.33 | [0.06, 0.79] |

The "seven failures" premise is dead. Each of those numbers was a single draw
from a distribution that escapes 10–33% of the time, and the β-parametrisation
hypothesis was built on top of them.

**`S5_c5_t:4` seed 3 reached token 0.9994 with parity resolved on both halves**
(0.9995 even / 0.9994 odd). That is a high-demand S₅ cell — frac(D=max) = 0.71 —
so `PLAN.md` Part 3b's training wall is refuted by counterexample, not merely
unsupported. See Part 3c for what replaces it.

**Error #7, added to §1:** `par[pred]` indexes a 120-entry parity table with an
argmax over a 127-token vocabulary. A model on the plateau can predict a special
token; on CUDA the out-of-bounds read is a device-side assert that kills the
process and every run queued behind it. The bug was latent in the original code,
which only evaluated 3 batches; fixing the truncated evaluation (error #2)
raised exposure ~5× and triggered it. Fixed, with self-test T5b, verified to
fail against the pre-fix code.

**Outstanding:** `S5_c5_t:4` seeds 8–10 (4–7 done: 0.9933, 0.0468, 0.0510, 0.9581
— 3 escapes of 7 so far, and both escapes resolve parity on *both* halves).

### Follow-up, 14 September: a second failure mode, and a wrong hypothesis of mine

`S5_c5_t1:4`, ten runs, constant LR:
`0.2453 0.3807 0.3966 0.4070 0.4750 0.4941 0.5376 0.6169 0.6939 0.7631`

**Not bistable.** The escape-fraction framing — which I had just finished writing
into three documents — does not apply to this cell, and the "4/10" a 0.5
threshold produces is an artifact of the cutoff. `exp_beta_param.py` now warns
when a distribution is not bimodal instead of printing that fraction silently.
**Error #8**, same class as the rest: a summary statistic applied outside the
regime it was built for.

Two claims in `PLAN.md` die outright, and one hypothesis of mine was simply the
wrong shape:

- **"Under-training is ruled out"** — false. 250k steps + cosine + 3× data gave
  0.5661; ten 20k constant-LR runs gave a median of 0.4845 and a best of 0.7631.
  12× the compute bought nothing.
- **"Models stall at the A₅ ceiling (0.545)"** — refuted by the parity split,
  which is only meaningful on S₅-generating cells and this is one. The best run
  resolves *odd* targets at 0.7199, and the even/odd gap shrinks as runs improve.
  0.5661 ≈ 0.545 was a coincidence we read a mechanism into.
- **My cosine-confound hypothesis** — not confirmed, not cleanly refuted, wrong
  question. It assumed this cell jumped and that LR decay suppressed the jump.
  There is no jump. Generalising the escape structure from four cells to a fifth
  was the same move as generalising a noise floor from one cell to all of them.

---

## 5b. Superseded plan text (kept for the record)

### Pending: the escape census

Running as of 13 Sept 10:5x UTC, ~7 h:

```bash
python work/exp_beta_param.py \
  --cells A5_c5:2,A5_c5_dt:2,A5_c5_dt:4,S5_c5_t:4 \
  --modes sigmoid:0 --seeds 1,2,3,4,5,6,7,8,9,10 \
  --out results/escape_census.json
```

Stock architecture only. It replaces four single-observation "failures" with
four measured escape probabilities.

**What it settles.** Whether these cells are reliably broken or bistable. A cell
showing 0 escapes in 10 runs is broken with `q < 0.25` at p = 0.056 — one run
short of the conventional threshold, so add 1–2 runs to any cell that returns
0/10 before calling it reliably broken.

**What it does not settle.** Whether any parametrisation helps. Distinguishing a
25% escape rate from 75% at p<0.05 with 80% power needs about **20 runs per
arm**; distinguishing 25% from 50% needs far more. Comparing β modes is a
bigger experiment than we have run so far, and it should not be attempted until
the census says which cells are worth comparing on.

---

## 6. Corrections to apply to `PLAN.md`

To be applied once the census returns, so they can carry real numbers:

1. **§5.3** — scope the noise floor to stable cells; add the bistable figure and
   the fixed-seed 0.9598/0.1177 observation.
2. **Part 3b and the trainability table** — mark every cell as a single draw,
   add the sensitivity table from §3, and downgrade "resource-independent,
   reproducible, and not reported anywhere in the literature" to the contrast
   that actually survives.
3. **§5.1** — flag mid-range entries as single draws from possibly bistable
   cells.
4. **Part 1 status board** — the row "High-demand regime is trainable at all:
   **no**" becomes "unmeasured; 0 of 4 single draws".
5. **Part 9 (errors caught)** — add rows 1–6 from §1 above. Two of them were
   caught by a reader rather than a test, which is the pattern that section
   exists to record.

---

## 7. Rules this adds to the working agreement

- **Escape fraction, never a mean.** On bimodal cells a mean describes no run
  that happened. Report `k/n` and the sorted list.
- **Seed is not the unit of replication** where kernels are non-deterministic.
  Hold the data seed fixed, vary the run, and report over runs.
- **A cell is "reliably broken" only at n ≥ 11** with zero escapes.
- **Controls come from the same invocation.** Never compare against a number
  from another script, another day, or another file.
- **Every derived statistic gets a known-answer test before it enters a
  document**, and the test must be shown to fail against the wrong
  implementation. A test that has only ever passed has demonstrated nothing.
