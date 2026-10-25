# Adaptive-Order DeltaProduct — Master Plan and Record

**Version 8** · 8 October 2026 · supersedes v1–v7 entirely (8: stopping rule applied, main deliverable redefined — Part 7)
Utkarsh Shukla · B.Tech AI, NITK Surathkal · Guide: Anand Kumar M

This is the single living document for the project. It carries the full record
from the first day, the current scientific position with evidence, what is
proved versus measured versus open, and the plan forward. Where an earlier
version of this document said something we have since disproved, the correction
is recorded rather than quietly deleted.

---

## Part 0 — What this project is

A year-long major project in core sequence-model architecture research. Not an
application. The goal is to find a real gap in the linear-attention /
state-tracking line of work, propose a mechanism that closes it, and publish.

The vehicle is **DeltaProduct** (Siems, Carstensen, Zela, Hutter, Pontil,
Grazzi — NeurIPS 2025, arXiv 2502.10297). DeltaProduct builds each recurrent
state-transition matrix as a product of `n_h` generalized Householder factors,
where `n_h` is a **fixed global hyperparameter**. Our mechanism makes that count
**per-token and learned**.

**Where we actually are:** the mechanism works and is implemented, but the more
valuable finding turned out to be a correction to the *cost model everyone in
this line of work has been assuming*, including DeltaProduct itself and
including us until 11 September. Part 3 is that correction. It is the strongest
thing we have and it is ours.

**Update, 8 October.** The learned gate now works on the toy task: trained on the
same hard gates evaluation uses, with the budget penalty from the first step, it
reaches the exact allocation (1 factor per transposition, 4 per 5-cycle, accuracy
≥ 0.9986) in **13 of 20** runs. That is short of the 15/20 reliability bar set in
advance, so — by that rule — the per-token gate is no longer the main
deliverable. It becomes the *instrument*: the main deliverable is a layer whose
order is learned by the gate and then built at that size, which turns the saving
into measured wall-clock time with standard kernels. Per-token savings stay as
the extension (they need a ragged kernel). Part 7 has the plan.

---

## Part 1 — Status board

| | Status | Evidence |
|---|---|---|
| Environment, GPU build, full reproduction pipeline | **done** | RTX 4050 6 GB / WSL |
| DeltaProduct S₃ reproduction | **done** | matches published 1/0 pattern |
| DeltaProduct S₅ reproduction | **done** | token 0.864 @ len 512, trained @ 128 |
| float64 oracle for the algebra | **done** | all 120 S₅ elements, 4.4e-16 |
| Halting gate, wired into the real kernel | **done** | 7/7 tests, gradient verified |
| `K ≥ ℓ` is forced by rank | **proved** | `exp_beta_reachability.py` E1 |
| The parity wall (β pinned ⟹ `K ≡ ℓ mod 2`) | **proved** | E2/E3, all 120, 1.2e-15 |
| No wider representation of S₅ helps | **proved** | E4, 10 reps, regular rep pays 96 |
| **Demand ≠ transposition length** | **proved** | `exp_minimal_order.py` V1–V5 |
| Capability at matched compute | **measured**, 3 of 5 points | sweep p = 0.05 / 0.1 / 0.25 |
| Group-closure prediction | **supported** (5 alphabets, perfect split at n_h=3) | `results/group_closure_results.md` |
| Threshold pinned at n_h=4 for S₅ | **no** — fails at every order, 250k steps | `results/gc_s5t1_long.json` |
| High-demand regime is trainable at all | **yes** — `S5_c5_t:4` reached token 0.9994 / seq 0.9415, parity resolved on both halves. The "wall" was a sampling artifact | `results/escape_census.json`; Part 3c |
| Every "failed" cell is bistable, not broken | **measured**, 33 runs | escape rates 0.10–0.33 at 20k steps; Part 3c |
| "Under-training ruled out" (250k run) | **false** — 250k+cosine+3× data scored 0.5661, the 70th percentile of ten 20k constant-LR runs (best 0.7631) | `results/escape_s5t1.json`; Part 3c |
| "Models stall at the A₅ ceiling (0.545)" | **refuted** — best run resolves *odd* targets at 0.7199; the gap shrinks as runs improve | parity split, Part 3c |
| Two distinct failure modes, not one | **measured** — `S5_c5_t:4` bistable and solves; `S5_c5_t1:4` grinds continuously 0.25–0.76 | Part 3c |
| Learned gate (trained end-to-end) | **first evidence**, not established — the gate closed on **1 of 3 seeds**, at every γ | `results/gate_train_p10.json`; Part 3d |
| Where it closes, the gate recovers the oracle allocation | **measured** on that one seed — cost 1.298 vs oracle 1.298, mae 0.00, corr 1.00 over 256 000 tokens, at both γ = 0.03 and γ = 0.1 | Part 3d |
| Gate closure is itself bistable, independently of accuracy | **measured** — accuracy 0.98–1.00 on every arm including the ones where cost stayed at exactly 4.000 | Part 3d |
| Learned gate, penalty-weight sweep (10 runs) | **measured** — 3 of 10 runs recover the demand structure at 2.199 factors/token, token 0.9998; no run under-allocates | `results/gate_gamma_sweep.json`; Part 3e |
| Learned gate across all configurations | **measured** — 7 of 49 runs recover the structure, from 4 distinct seeds; best allocation 1.2985 (exact) on one seed of the earliest configuration only | Part 3e |
| **Main deliverable (8 Oct)** | **redefined** — layer sized by the learned gate, measured wall-clock saving; per-token savings become the extension | Part 7, revised plan |
| Learned gate, hard gates + penalty from step 1 (7 Oct) | **measured** — **13 of 20** seeds reach the exact 1.2985 with token ≥ 0.9986 [0.43, 0.82]; all 7 failures under-allocate 5-cycles and never learn the task. Stopping-rule threshold (15/20) **not met** | `results/gate_early_B20.json`; Part 3e |
| Gate must be evaluated in float32 | **measured** — bf16 sigmoid saturates above logit 6.2364 with exactly zero gradient; fp32 threshold 17.33 | `results/gate_fp32gate.json`; Part 3e |
| State tracking in natural language (Boxes) | **measured**, fixed order only — DeltaProduct n_h=2 reaches 0.968 exact match after one operation vs 0.312 for equal-size attention; hybrid does not help | `results/boxes_hybrid.json`; Part 3f |
| Training time by fixed order | **measured** (found 11 Oct in existing files) — 0.45 / 0.58 / 0.79 / 1.00 of the 4-factor time for orders 1–4 | §5.1 result files; Part 7, review of 11 Oct |
| Real wall-clock saving (ragged expansion) | **not done** | adaptive costs are notional; ragged expansion not built |

---

## Part 2 — What we have established

### 2.1 The reproduction is sound

We rebuilt DeltaProduct from the authors' code and reproduced both published
state-tracking results on a 6 GB laptop GPU. S₅ with `n_h = 4`, one layer,
12 heads × 32, 500k samples, 100 epochs (7h45m): **token accuracy 0.864 at
length 512 after training at length 128**, with whole-sequence accuracy perfect
through position ~121 and still 0.50 at ~250. Graceful length extrapolation,
matching the paper qualitatively. Earlier S₅ failures were under-resourcing
(data, width, optimizer steps), not architecture.

This matters because every comparison we make is against this baseline. If the
baseline were broken, nothing downstream would mean anything.

### 2.2 What β is, and the two hard bounds

A generalized Householder factor is `H(β,k) = I − β·kkᵀ` with `‖k‖=1`,
`β ∈ [0,2]`. It touches only direction `k`, scaling that component by `(1−β)`:
β=0 is the identity, β=1 the delta-rule projection, β=2 a reflection. A
transposition *is* a reflection, which is the origin of "a permutation of
transposition length ℓ needs ℓ factors."

**Bound 1 — rank.** Each factor differs from `I` by a rank-1 matrix, so
`rank(A − I) ≤ K`. And `rank(P − I) = n − cycles(p) = ℓ(p)`. Therefore `K ≥ ℓ`
is forced **for every β**, free or pinned. Verified over d ∈ {5,8,16},
K ∈ 0..6.

**Bound 2 — parity, and why the gate needs a learnable β.** With β pinned at 2,
`det(A) = (−1)^K` while `det(P) = (−1)^ℓ`, so only `K ≡ ℓ (mod 2)` is
reachable. Constructively: padding ℓ factors up to K requires building
identities out of the spare factors, and there are only two ways — a *pair* of
identical reflections (costs 2), or a *single* closed factor β=0 (costs 1).
Pinned β has only the pair, hence even padding only.

Verified by explicit construction on **all 120 elements at every K**, worst
reconstruction error **1.17e-15**:

```
LEARNABLE β ∈ [0,2]              PINNED β = 2
       K=0  K=1  K=2  K=3  K=4          K=0  K=1  K=2  K=3  K=4
L=0      .    .    .    .    .     L=0     .    x    .    x    .
L=1      x    .    .    .    .     L=1     x    .    x    .    x
L=2      x    x    .    .    .     L=2     x    x    .    x    .
L=3      x    x    x    .    .     L=3     x    x    x    .    x
L=4      x    x    x    x    .     L=4     x    x    x    x    .
```

**Consequence for the implementation:** the gate must be able to drive β to 0,
not mask a fixed β. Our patch multiplies the gate into β *after* the
`sigmoid()*2`, so it was already correct — but now we know why it had to be
rather than having guessed.

**Consequence for the claim:** since the gate can never go below the demand,
the adaptive cost is a *hard floor*, not a convenient number. Our claimed
operating point is optimal, not merely achievable.

---

## Part 3 — The central finding: the demand is not the transposition length

### 3.1 The result that forced it

From the matched-compute sweep at **p = 1.0**, where every input token is a
5-cycle (ℓ = 4):

```
fixed n_h=1   token 0.0648
fixed n_h=2   token 0.1207
fixed n_h=3   token 0.9994   seq 0.9535    <-- solves it
fixed n_h=4   token 0.9995   seq 0.9600
```

Three factors track a sequence of 5-cycles. But Bound 1 says three factors
cannot represent one. Both results are solid, so a joining assumption is wrong,
and it is this one: *that the matrix the layer must apply is the permutation
matrix of the token.* It is not. The layer may apply `ρ(g)` for **any faithful
representation ρ of the group it needs to track**, and ρ is its choice.

### 3.2 The explanation, verified

Five-cycles are even, so an alphabet of 5-cycles generates **A₅, not S₅**. And
A₅ — unlike S₅ — has a **3-dimensional faithful representation**: it is the
rotation group of the icosahedron. In SO(3) every non-identity rotation fixes
exactly its axis, so `rank(R − I) = 2`, and by Cartan–Dieudonné every 3-D
rotation is a product of exactly **two** reflections.

`work/exp_minimal_order.py` builds this group from geometry and verifies every
step:

- 60 distinct rotations, closed, orthogonality error 1.4e-16
- element-order profile `{1:1, 2:15, 3:20, 5:24}` — A₅'s
- the isomorphism exhibited explicitly, via the action on the 5 inscribed
  octahedra: faithful, all images even, onto all 60, homomorphism verified
- **every non-identity element has rank 2**, including the 5-cycles
- each factors as exactly 2 Householder reflections, worst error **1.31e-15**

```
element type              count   L (perm rep)   rank in 3-D
identity                      1              0             0
double transposition         15              2             2
3-cycle                      20              2             2
5-cycle                      24              4             2
```

### 3.3 The corrected cost model

> **D(g) = min over faithful representations ρ of ⟨alphabet⟩ of rank(ρ(g) − I)**

| group | \|G\| | min faithful dim | D over non-identity | ℓ range | adaptivity can help? |
|---|---|---|---|---|---|
| S₃ | 6 | 2 | 1, 2 | 1–2 | yes |
| S₄ | 24 | 3 | 1, 2, 3 | 1–3 | yes |
| **A₄** | 12 | 3 | **2 only** | 2 | **no** |
| **A₅** | 60 | 3 | **2 only** | 2–4 | **no** |
| S₅ | 120 | 4 | 1, 2, 3, 4 | 1–4 | yes |

Three things follow, and the third is the interesting one.

**(a) ℓ is right for symmetric groups.** In the standard (n−1)-dim rep of Sₙ,
`rank(ρ(g) − I) = ℓ(g)` exactly. So our oracle is correct wherever the alphabet
generates S₅ — the p = 0.05 / 0.1 / 0.25 results stand as measured.

**(b) ℓ is wrong for A₅**, by a factor of 2 on 5-cycles. Our p=1.0 "negative
control" charged the oracle 4 when 2 would do. The control fired for the right
reason (no gap exists at p=1) but with the wrong number, and `fixed3` solving is
the tell. This also retires an anomaly open since the start of the project:
**A₅ trains at n_h=2 because 2 is genuinely sufficient**, not because the theory
was loose.

**(c) The demand is a property of the alphabet, not the token.** This is the
testable, counterintuitive prediction:

> Adding **one transposition** — the cheapest token that exists, ℓ=1 — to an
> alphabet of 5-cycles closes it over S₅ instead of A₅, destroys the
> 3-dimensional representation, and **raises the cost of every 5-cycle from 2
> to 4**. An easy token makes the task harder.

No per-token difficulty model can produce that. If it holds, per-token
difficulty is the wrong frame — and that is a correction to DeltaProduct's own
framing, not just to ours.

**Our data already contains the natural experiment**, though not cleanly:

| | alphabet | generates | `fixed3` token acc |
|---|---|---|---|
| p = 0.25 | 10 transpositions + 24 five-cycles | S₅ | **0.5516** |
| p = 1.0 | 24 five-cycles | A₅ | **0.9994** |

Removing the *easy* tokens took `fixed3` from 0.55 to 0.9994 on the same
5-cycles. Suggestive, not decisive — the mixture and the group both changed.
`work/exp_group_closure.py` separates them.

### 3.4 The honest cost to us

Where D is constant over non-identity elements — A₄ and A₅ — **adaptive order
cannot help at all**. The script prints this as a failure of our own mechanism,
by design. We can now predict *before running* which alphabets our method can
help on, which is a stronger position than claiming it always helps.

---

## Part 3b — The high-demand training wall (13 Sept)

`S5_c5_t1` (24 five-cycles + one transposition, 96% high-demand tokens) was run
at **250 000 steps, warmup + cosine, 300k samples** — 10× the steps and 3× the
data of every earlier cell.

| n_h | 20k steps | 250k + schedule |
|---|---|---|
| 2 | 0.0904 | **0.3617** |
| 3 | 0.1094 | **0.4828** |
| 4 | 0.4851 | **0.5661** |

**It still does not solve at any order.** So the failure is not under-training,
which was the standing explanation and is now dead.

Two things were bought for the six hours:

1. **The ordering is now clean.** At 20k the numbers were 0.09 / 0.11 / 0.48 —
   no order at all. At 250k they are monotone in n_h, which is what capacity
   predicts. The measurement is trustworthy now even though the task is not
   solved.

2. **They stop at the A₅ ceiling.** A model that tracks A₅ but cannot resolve
   the parity bit scores ≈0.545 (computed, not fitted). n_h=4 landed at 0.5661.
   The models learn the even structure — affordable at D=2 — and cannot learn
   the single bit separating S₅ from A₅, even where they provably have the
   capacity.

### The wall, stated

Across six independent runs the architecture solves S₅ tasks easily when
transpositions dominate (0.9999 at p = 0.05–0.25) and A₅ tasks easily (0.999).
It fails whenever **most tokens require the maximum order in S₅**:

| frac(D = max) | ≤ 0.25 | ≥ 0.50 |
|---|---|---|
| trains | 6 of 7 | **0 of 4** |

> ## RETRACTED — 13 September, by measurement
>
> This paragraph previously read *"This is resource-independent, reproducible,
> and not reported anywhere in the literature."* **The wall does not exist as
> stated.** The escape census (`results/escape_census.json`, 33 runs of the
> unmodified architecture) shows every cell we called a failure is *bistable*:
> it escapes the plateau some fraction of the time, and each historical number
> was one unlucky draw.
>
> | cell | historical (n=1) | escapes / runs | rate | 95% CI |
> |---|---|---|---|---|
> | `A5_c5:2` | 0.1259 | **3 / 10** | 0.30 | [0.11, 0.60] |
> | `A5_c5_dt:2` | 0.0443 | **2 / 10** | 0.20 | [0.06, 0.51] |
> | `A5_c5_dt:4` | 0.0429 | **1 / 10** | 0.10 | [0.02, 0.40] |
> | `S5_c5_t:4` | 0.0706 | **1 / 3** † | 0.33 | [0.06, 0.79] |
>
> † the census crashed at run 34 of 40 on an unrelated indexing bug; seeds 4–10
> of this cell are outstanding.
>
> **The decisive observation is `S5_c5_t:4` seed 3.** That is a *high-demand S₅
> cell* — 24 five-cycles and 10 transpositions, so frac(D=max) = 0.71, squarely
> in the "0 of 4" column. It reached **token 0.9994, sequence 0.9415**, with
> parity accuracy **0.9995 on even targets and 0.9994 on odd**. It resolves the
> parity bit completely.
>
> So two claims die together:
>
> 1. **The wall.** A high-demand S₅ cell trains to 0.9994. "0 of 4" was a 32%
>    coincidence, exactly as the sensitivity table in `AUDIT.md` §3 predicted.
> 2. **The A₅ ceiling reading below.** The architecture *can* learn the bit
>    separating S₅ from A₅ at n_h=4. The 0.5661 in the table above is not a
>    capability ceiling; it is a run that did not escape.
>
> **What the phenomenon actually is.** Not a representational limit but a
> low-probability escape from a loss plateau. Escapes are abrupt and scattered
> in time — across the census they occurred before step 5 000, between 5–10k,
> 10–15k and 15–20k, with some runs still descending at the 20 000-step cutoff.
> The 20 000-step escape rates above are therefore *lower bounds*.
>
> **This is better news than the wall was.** An optimisation-basin problem is
> more tractable than a representational one, and "why is the escape rare, and
> can it be made reliable" is a question worth a project. See Part 3c.

**And it is good news for this project.** Adaptive order is only useful when
demand is *heterogeneous* — a few expensive tokens among many cheap ones — and
that is exactly the regime that trains. The broken regime is the one where
adaptivity has nothing to offer anyway.

---

## Part 3c — The plateau escape (13 September, replaces the wall)

### What we measured

33 runs of the **unmodified** architecture across four cells, 20 000 steps,
constant learning rate, data held fixed and only the run varied. Every cell is
bistable. Escape from the plateau is abrupt: the loss sits at chance (≈3.6 for
A₅ cells, ≈4.0 for S₅ cells) for thousands of steps and then falls by two orders
of magnitude within one logging interval.

Escape times observed, bracketed by the logging interval:

| cell | escapes at |
|---|---|
| `A5_c5:2` | <5k, <5k, 5–10k |
| `A5_c5_dt:2` | 5–10k, 10–15k |
| `A5_c5_dt:4` | 15–20k |
| `S5_c5_t:4` | 10–15k |

### The hypothesis this suggests, and its test

If escape is a **memoryless waiting process** at rate λ per step, then
`P(escape by T) = 1 − exp(−λT)`, and the 20 000-step rates imply:

| cell | λ per step | median escape step | P(escape) @100k |
|---|---|---|---|
| `A5_c5:2` | 1.8e-05 | ~38 900 | 0.83 |
| `A5_c5_dt:2` | 1.1e-05 | ~62 100 | 0.67 |
| `A5_c5_dt:4` | 5.3e-06 | ~131 600 | 0.41 |
| `S5_c5_t:4` | 2.0e-05 | ~34 200 | 0.87 |

**This is a hypothesis, not a result.** It is fitted from seven escape events
and the memorylessness is untested. It makes one sharp prediction — *run longer
and most of these "failures" train* — and that prediction is cheap to falsify.

### The confound it exposes in our longest run

`S5_c5_t1:4` ran 250 000 steps with **warmup + cosine decay** and did not escape
(0.5661). Under the model above at the `S5_c5_t:4` rate, `P(no escape by 250k)
= 0.006`. Three readings are possible: that cell is genuinely much harder; the
memoryless model is wrong; or **the cosine schedule prevented a late escape by
decaying the learning rate to zero.**

The 20 000-step census runs held LR constant and could escape at any point. The
250 000-step run could not escape late. That makes the schedule a confound, not
a control — and it means **"under-training is ruled out" is no longer
established**, since the run was not under-trained but possibly decayed into the
plateau.

**The experiment that separates these:** `S5_c5_t1:4`, ten runs, **constant LR**,
20 000 steps (~2.1 h). If it escapes at a comparable rate, the schedule was the
problem and the longest run in the project was measuring its own optimiser.

### The answer (14 September) — the hypothesis was the wrong shape

Ten runs of `S5_c5_t1:4`, constant LR, 20 000 steps
(`results/escape_s5t1.json`):

```
0.2453  0.3807  0.3966  0.4070  0.4750  0.4941  0.5376  0.6169  0.6939  0.7631
```

**This cell is not bistable at all.** No run near 0.999, none near 0.04; the
outcomes fill the middle continuously. The escape-fraction framing does not
apply to it, and the "4/10" a 0.5 threshold produces is an artifact of where the
cutoff was put. `exp_beta_param.py` now detects this case and says so rather
than printing a misleading fraction.

So the project contains **two different failure modes** that had been lumped
together, and they track the fraction of transpositions in the alphabet:

| cell | transpositions | behaviour | when it succeeds |
|---|---|---|---|
| `S5_c5_t:4` | 10 of 34 (29%) | **bistable** — 0.047, 0.051, 0.958, 0.993, … | solves outright, parity resolved on both halves |
| `S5_c5_t1:4` | 1 of 25 (4%) | **continuous grind** — 0.25 to 0.76, loss still falling | never solves at this budget |

`S5_c5_t1` has to learn a 120-element group from data in which the only
parity-flipping generator appears in 4% of tokens. That is a data and curriculum
problem, not an architectural one.

### Three things this settles

**1. "Under-training is ruled out" is definitively wrong.** The 250 000-step run
— with cosine decay *and* 3× the data — scored **0.5661**. That is the 70th
percentile of these 20 000-step constant-LR runs, and the best of them reached
**0.7631**. Twelve times the steps and three times the data bought nothing
measurable. Whatever stopped that run, it was not a shortage of optimiser steps.

**2. The A₅ ceiling reading is refuted, by the diagnostic that finally applies.**
`S5_c5_t1` generates S₅, so it *has* odd targets and the parity split is real
here (unlike every A₅ cell, where it is undefined):

| run | even | odd | gap |
|---|---|---|---|
| seed 6 (best) | 0.7989 | **0.7199** | 0.079 |
| seed 1 | 0.7680 | 0.6046 | 0.163 |
| seed 5 (worst) | 0.3492 | 0.1201 | 0.229 |

A model trapped in A₅ scores ~0 on odd targets. **None of these do**; the best
resolves odd targets at 0.72. And the gap *shrinks* as runs get better, which is
gradual learning of the whole group, not a coset ceiling. The 0.5661 in Part 3b
was never "the A₅ ceiling at 0.545" — 0.5661 and 0.545 were a coincidence, and
we read a mechanism into it.

**3. The cosine hypothesis is neither confirmed nor cleanly refuted — it was the
wrong question.** It assumed this cell had the abrupt-escape structure and that a
decaying LR was suppressing a jump. There is no jump here to suppress. The
hypothesis was a reasonable inference from the other four cells and it did not
survive contact with this one.

### What is actually open

Why does `S5_c5_t1:4` plateau in the middle instead of solving, when
`S5_c5_t:4` — the same group, the same order, more transpositions — solves
outright? The obvious candidate is generator scarcity, and it is directly
testable:

- **Constant LR at 250k steps**, 3 seeds (~7.8 h). The exact comparison to the
  cosine run, with the schedule removed. Loss was still descending at 20k in
  5 of 10 runs.
- **Oversample the transposition** to 10–30% and see whether the cell moves from
  "grind" to "bistable". If it does, demand heterogeneity is not the variable
  that matters — generator coverage is.

---

## Part 3d — The learned gate, first end-to-end run (15 September)

Until this run, every adaptive number in this document came from the **oracle**
— a control condition handed the correct per-token order from ground truth. This
is the first experiment in which the halting gate received no supervision on the
allocation and had to infer it from the task loss and a budget penalty.

`work/exp_gate_train.py`, p = 0.10, 18 runs: three arms (`fixed4`, `oracle`,
`learned`) × four budget weights γ ∈ {0, 0.01, 0.03, 0.1} × three seeds.
20 000 steps each, penalty off for 4 000 steps then ramped over 4 000. Gates are
soft while training and **hard at evaluation**, so every cost below is an integer
factor count, not a sum of sigmoids.

### The headline, stated as a fraction

**The gate closed on 1 of 3 seeds.** On that seed it did not merely approach the
oracle allocation — it reproduced it:

| seed 1 | token | seq | cost | mae | corr | D=1 tokens get | D=4 tokens get |
|---|---|---|---|---|---|---|---|
| oracle | 0.9996 | 0.9770 | 1.298 | 0.00 | 1.00 | 1.000 | 4.000 |
| learned γ=0.03 | 0.9999 | 0.9910 | **1.298** | **0.00** | **1.00** | **1.000** | **4.000** |
| learned γ=0.1 | 1.0000 | 0.9970 | **1.298** | **0.00** | **1.00** | **1.000** | **4.000** |
| fixed4 | 0.9998 | 0.9885 | 4.000 | — | — | — | — |

mae is the mean absolute difference between spent and demanded order, over
**256 000 evaluated tokens**, and it is 0.00 — not a small number, zero to the
reported precision. Every low-demand token received exactly one factor and every
high-demand token exactly four. This matters because a gate can reach the right
*average* by closing uniformly, which would be worthless; the by-demand split is
the test that separates the two, and the script prints a FLAT warning when it is
not passed. Two independent γ values reached the same allocation.

The accuracy differences against the oracle (seq 0.9970 vs 0.9770) are inside the
noise floor of §5.3 and are **not** claimed as the gate beating the oracle.

### And on 2 of 3 seeds it did not close at all

Seeds 2 and 3: cost **exactly 4.000** at every γ, including γ = 0.1, with
corr 0.00 and D=1 tokens receiving all four factors.

Note what is *not* failing. Token accuracy is 0.98–1.00 on every arm in the
experiment, the non-closing ones included. **The task is learned in all 18 runs;
it is the gate that is bimodal.** That is a different phenomenon from the
plateau bistability of Part 3c, where accuracy itself was the bimodal quantity.

### The γ = 0 control is informative

γ = 0 installs the gate and never penalises it. On seed 1 that arm still reached
cost 3.369 with corr 0.43 (D=1 tokens averaging 3.30 against D=4 tokens at
4.00), so **the task loss alone had begun closing the gate on easy tokens before
any budget pressure was applied**. On seeds 2 and 3 the γ = 0 arm sat at 4.000
with corr 0.00. The seed split is therefore visible *before* the penalty exists,
which points away from the penalty weight as the cause.

### What this does not establish

- Not a speedup of any kind. Path A still expands all K factors and multiplies
  the disabled ones by zero. Cost here is **required work per token**.
- Not a reliable mechanism. One escaping seed out of three is first evidence,
  and the next run is aimed at making it reproducible, not at writing it up.
- Only one cell (p = 0.10) and only one task family.

### The suspected mechanism, and why γ is probably the wrong lever

γ = 0.1 left seeds 2 and 3 at cost *exactly* 4.000 — not 3.9, not 3.5. A gate
under insufficient pressure drifts down; a saturated one does not move. The
initialisation is the obvious suspect:

- `init_open_bias = 4.0` gives λ = σ(4) = 0.982, whose derivative is **0.018**
- `nn.init.zeros_(self.mlp[-1].weight)` makes the gate **input-independent at
  init** — every token receives the same order until that weight leaves zero

So the gate begins as a constant function sitting in the flat tail of a sigmoid,
and the budget penalty pushes only the bias, uniformly. Only the task loss can
create input-dependence, and on seed 1 it did. This is a hypothesis, not a
finding; the next run tests it directly and it can fail.

---

## Part 3e — The learned gate, penalty-weight sweep and reliability history (18–19 September)

Part 3d was three seeds in one configuration. This is the follow-up: the gate
trained end to end at p = 0.10 under two penalty weights, five runs each, data
held fixed and only the run varied (`results/gate_gamma_sweep.json`).

Ideal allocation on this cell is **1.2985** factors per token (90% of tokens
require 1, 10% require 4).

| γ | run | token | seq | n̄ | n̄ \| D=1 | n̄ \| D=4 | corr | under-alloc |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 1 | 0.99840 | 0.9405 | 3.9103 | 3.900 | 4.000 | 0.104 | 0.000 |
| 0.1 | 2 | 0.99985 | 0.9870 | **2.1990** | **2.000** | 4.000 | **1.000** | 0.000 |
| 0.1 | 3,4,5 | ≥0.99896 | — | 4.0000 | 4.000 | 4.000 | 0.000 | 0.000 |
| 0.3 | 1 | 1.00000 | 0.9995 | 3.8206 | 3.801 | 4.000 | 0.155 | 0.000 |
| 0.3 | 2 | 0.99980 | 0.9805 | **2.1990** | **2.000** | 4.000 | **1.000** | 0.000 |
| 0.3 | 3 | 0.99956 | 0.9640 | **3.0995** | **3.000** | 4.000 | **1.000** | 0.000 |
| 0.3 | 4,5 | ≥0.99978 | — | 4.0000 | 4.000 | 4.000 | 0.000 | 0.000 |

**What holds.** Three of ten runs separate the two requirement classes exactly,
with no supervision on the allocation. **No token is under-allocated in any run
of this sweep**, and 5-cycles receive all four factors in every run. (Not
universal: seed 16 of the 2 October logit-wall experiment under-allocates 3–5%
of tokens, in both arms, in runs where the task itself was not learned.) Token accuracy is
≥ 0.998 everywhere in this sweep, including the runs where the gate never
closes, so here adding the gate is accuracy-neutral. (Not across all 49 runs:
five fall below 0.998, the lowest `gate_fp32gate` seed 7 at 0.7944 — a run that
separated the classes at 3.0995 while losing the task, and is correctly not
counted as recovered.) The best runs are both cheaper and more accurate
than `fixed3` on the same cell (3 factors, 0.9696).

**What does not hold.** 3/10 is not a reliable mechanism, and the best runs stop
at 2.199 rather than the exact 1.2985 — swaps get two factors instead of one.

### The reliability history, across every configuration tried

| configuration | γ | runs | recovered | lowest n̄ (token) |
|---|---|---|---|---|
| pre-repair init, seeds 1–3 (`gate_train_p10`) | 0, 0.01, 0.03, 0.1 | 12 | 2 | **1.2985** (0.99990) |
| pre-repair init, seeds 4–10 (`gate_bias40_more`) | 0.03 | 7 | 1 | 2.1990 (0.99961) |
| gate re-initialised (`gate_fixedinit`) | 0.03 | 10 | 0 | — |
| float32 halting (`gate_fp32gate`) | 0.03 | 10 | 1 | 2.1990 (0.99984) |
| penalty-weight sweep (`gate_gamma_sweep`) | 0.1, 0.3 | 10 | 3 | 2.1990 (0.99985) |
| **total** | | **49** | **7** | |

The **exact** allocation of 1.2985 — one factor to every transposition, four to
every 5-cycle — was reached only in the earliest configuration, twice. Nothing
since has gone below 2.199. Either one of the later "fixes" is a regression, or
those two cells were lucky draws. Settling this is the first item in the queue,
and it gates the value of everything built on top.

**Corrections to the table above (1 October, from the result files).**

- **Order and label.** The rows are now chronological. `gate_bias40_more` ran
  *before* the initialisation repair, not after it, and was not a bias sweep in
  effect: its log header printed "init bias 4", but its measured mean order at
  step 1 is **0.933–0.945** on all seven runs — the 0.9375 that HuggingFace
  `post_init` leaves (bias 0, weights N(0, 0.02)). Every repaired run starts at
  3.81 (bf16) or 3.82 (fp32).
- **`gate_train_p10` did not start open either.** That script logged n̄ as 0.000
  during warmup, so this was first inferred: its step-1 losses for seeds 1–3
  (4.8671 / 4.8718 / 4.8681) differ from the open-gate runs on the same seeds
  (4.8658 / 4.8721 / 4.8686). **Confirmed 1 October by fingerprint:** rebuilding
  with `--gate_init hf_default` reproduces all six `gate_train_p10` step-1 losses
  (oracle and learned, seeds 1–3) to the stored 4 decimals, with the gate
  starting at n̄ 0.935–0.946. The same rebuild reproduces all seven
  `gate_bias40_more` seeds exactly on step-1 loss, n̄ and gate |W| (20 of 20
  lines MATCH). The pre-repair model is therefore rebuilt exactly at step 1.
- **Runs are not independent outcomes.** The 7 recovered runs come from 4 seeds:
  `gate_train_p10` seed 1 (twice), `gate_bias40_more` seed 7, seed 2 under the
  float32 code (three times, at γ = 0.03, 0.1, 0.3, with near-identical
  trajectories), and seed 3 at γ = 0.3 (the same seed stayed at 4.000 at
  γ = 0.1). Outcome is strongly seed-correlated but not seed-determined, and no
  configuration has yet been run twice identically.
- **The seed that reached 1.2985 has never been re-run with the initialisation
  it was trained from.** Seeds 1–3 were re-run only with the repaired, open gate.
- **Saturation before the penalty.** In 36 of the 37 runs that logged it, the
  gate is fully open (n̄ ≥ 3.99, output |W| 15–23) by step 2000, inside the
  penalty-free warmup — including the pre-repair runs that started at 0.94. The
  exception is `gate_bias40_more` seed 7 (2.82 at step 2000), which is also that
  configuration's only success. In bf16 (`gate_fixedinit`, and 5 of 7
  `gate_bias40_more` runs) |W| is then bit-identical from step 2000 to 20 000.
- **Gate weight norm at the pre-repair start.** `gate_bias40_more`'s step-1
  output |W| is 6.3–7.0. A back-of-envelope estimate of 4.1 suggested a
  mismatch; it was wrong. The default construction measured directly
  (`check_gate_init.py` sections A and D) gives 6.66, in the same range.

### Precision: why the gate is evaluated in float32

`work/check_gate_precision.py`. In bfloat16 a sigmoid rounds to exactly 1.0 for
every logit above **6.2364**, with gradient exactly zero; the fp32 threshold is
17.3287. A gate driven into that region is frozen open and the budget penalty
can never close it. Measured directly: with logits set to 10 and 300 steps of
budget-only optimisation, bf16 gives gradient 0 and n̄ stays at 4.0000, while
fp32 closes from 3.9995 to 0.0625. With small logits both precisions agree
(gradients 9.359 vs 9.304), so the effect is specific to saturation. The gate is
therefore computed in float32; the recurrence stays in bf16, as the kernels
require.

### Gate initialisation

The gate's output layer starts with zero weights and bias 4.0, giving
λ = σ(4) = 0.982 and an initial effective order of 3.8233 for every token: the
gate starts open and input-independent, and must learn to close. Verified in the
training code path — the model class's default initialisation gave 0.9376
instead, which would start the gate almost shut. **Both runs made before this
repair (`gate_train_p10`, `gate_bias40_more`) started from that default**, so
the only exact allocation in the project came from a gate that started almost
shut, not open. `exp_gate_train.py --gate_init hf_default` rebuilds that start.

### Rebuilding the earliest configuration (1–2 October)

`results/gate_hfinit_p10.json`. The pre-repair start (`--gate_init hf_default`),
p = 0.10, 20 000 steps, seeds 1, 2, 3 at γ ∈ {0.03, 0.1}, then seed 1 again at
both γ. All 8 runs match `gate_train_p10` at step 1. Warmup logged every 250 steps.

| seed | γ | n̄ (original) | n̄ (this run) | n̄ \| D=1 | token |
|---|---|---|---|---|---|
| 1 | 0.03 | 1.2985 | **2.1990**, repeat **1.4778** | 2.0, repeat 1.199 | 0.99996, repeat **0.8937** |
| 1 | 0.1 | 1.2985 | **1.2985**, repeat **2.1990** | 1.0, repeat 2.0 | 0.99996, 0.99993 |
| 2 | 0.03, 0.1 | 4.0, 4.0 | 4.0, 4.0 | 4.0 | ≥ 0.9995 |
| 3 | 0.03, 0.1 | 4.0, 4.0 | 4.0, 4.0 | 4.0 | ≥ 0.9996 |

**What it settles.**

1. **There is no regression.** The exact allocation is reproduced (seed 1,
   γ = 0.1), and the *identical* configuration repeated gives 2.199. So 1.2985
   and 2.199 both come from one configuration; the later changes did not raise a
   floor. Item 1 of Part 7 closes on its failure condition, in the form "the
   earlier cells were partly draws".
2. **Whether the gate closes is set by the seed; where it closes is not.** Seed 1
   separated the classes in 4 of 4 runs; seeds 2 and 3 in 0 of 8 across both
   configurations. Among seed 1's runs the closure level varied between identical
   repeats (1.2985 / 2.199; 2.199 / 1.4778). So the seed fixes the outcome's
   class and run-to-run noise picks the level.
   **Corrected 3 October:** not in general. Six seeds that all closed on 2 October
   (12, 13, 15, 25, 29, 30) closed again in only 4 of 12 runs of the same
   configuration up to step 12 000 (`gate_hard_6seeds.json`) — below the base
   rate. Seeds 1–3 looked
   determined; across seeds 11–30 the seed is at most a weak predictor.
3. **The class is decided in the first ~1 000 steps, before any penalty.** Seeds
   2 and 3 reach n̄ = 4.00 by step 500–1 000 and their gate |W| is then
   bit-identical to step 20 000 (bf16 saturation, Part 3e). Seed 1 never fully
   opens: n̄ sits at 2.7–3.3 through the warmup, and drops to its final level
   within 2 000 steps of the penalty switching on. Across all 45 runs that logged
   their warmup, no bf16 run that was fully open by step 2 000 ever closed.
   **Corrected 3 October:** one has since — seed 12 (hard-gate arm) was fully
   open at step 2 000 with 91% of probabilities saturated, lost saturation by
   step 6 000 as the gate's inputs drifted (0.35), then closed. Saturation is
   usually permanent, not always.
4. **One run lost accuracy late.** Seed 1, γ = 0.03, repeat: task loss rose from
   0.004 to 0.012 and n̄ from 1.43 to 1.88 at the final logged step; evaluation
   token accuracy 0.8937. One logged point, so no trend is claimed; it fails the
   ≥ 0.99 rule and is not counted as recovered.

**Consequence.** The lever is not the penalty weight or the precision but the
gate saturating open during the penalty-free warmup. That is a property of our
gate (an unbounded sigmoid in bf16 with nothing opposing it for 4 000 steps),
not of DeltaProduct.

### The logit wall, 20 seeds per arm (2 October)

`results/gate_wall_seeds.json` (seeds 11–30, γ = 0.1, pre-repair start, each seed
with and without a wall at |z| = 5) and `results/gate_wall_anchor.json` (wall on,
seeds 1–3). Driver `work/run_gate_wall.sh`. Single run per seed and arm.

| | no wall | wall at 5 |
|---|---|---|
| any halting probability saturated (exactly 1.0) | 20 / 20 | **0 / 20** (and 0/3 anchors) |
| gate closed with the class structure (corr = 1.00) | **10 / 20** [0.30, 0.70] | 8 / 20 [0.22, 0.61] |
| … and token accuracy ≥ 0.99 (separated) | 4 / 20 [0.08, 0.42] | 1 / 20 [0.01, 0.24] |
| exact allocation 1.2985 with accuracy ≥ 0.99 | 1 / 20 | 0 / 20 |
| task loss > 0.05 at step 4 000 (slow to learn) | 1 / 20 | 5 / 20 |

**1. The wall removes saturation and does not improve closure.** It does what it
was built to do — no probability ever reaches 1.0, largest logit ≤ 6.00 — yet
the gate stays open in 11 of 20 seeds, and the anchors behave as before (seed 1
closes to 1.2985; seeds 2 and 3 do not). It also slows task learning (seed 17
never learns the task). By the criterion written before the run, saturation is
not *the* cause of non-closure, and the wall is dropped. Control
(`gate_wall_nobudget.json`, wall on, γ = 0, seeds 11–13): 0/3 closed, soft order
held at 3.922 throughout, so the wall does not close the gate on its own in
training — the drag seen in self-test T17d needs the task loss to be absent.

**2. Without the wall, saturation sets the level in every run that closed — but
being unsaturated is not enough to close.** Saturated fraction at step 4 000, when
the penalty switches on:

| saturated fraction at step 4 000 | predicted if exactly the kept factors saturate | final n̄ |
|---|---|---|
| 0.323–0.327 (4 runs) | 0.9·1/4 + 0.1 = 0.325 | 1.2985 |
| 0.572 (1 run) | 0.9·2/4 + 0.1 = 0.550 | 2.199 |
| 0.750–0.797 (5 runs) | 0.9·3/4 + 0.1 = 0.775 | 3.0995 |
| 1.000 (7 runs) | 1.0 | 4.000 |
| 0.954, 0.889 (seeds 19, 28) | — | **4.000** |

(Seed 16, which did not learn the task, is omitted.)

In all 10 runs that closed, the saturated fraction was already at its final
level by step 1 000 — before the penalty — and the factors kept open match it.
(This matches fractions; which individual factors saturated was not logged.)
But seeds 19 and 28 had 5–11% of probabilities unsaturated when the penalty
started, and 22–45% by the end, and closed none of them. So saturation is a lock,
not the only one — the same conclusion as the wall arm, where nothing saturated
and 11 of 20 seeds still kept every factor open. Something else formed in the
penalty-free warmup holds them open; a task solution already built on all four
factors is the obvious candidate, untested.

**3. A new problem, and probably ours: accuracy is lost at evaluation, not in
training.** Of the runs that closed and then scored below 0.99, 12 of 13 end
training with a task loss of 0.003–0.028 (no wall: 6/6; wall: 6/7, the seventh
collapsed late), the same as runs that evaluate at 0.999. E.g. seed 12, no wall:
final training loss 0.0058, evaluated token accuracy **0.652**. Training uses
soft gates and evaluation hard {0, 1} gates; soft n̄ at the end exceeds the hard
cost (1.333 against 1.2985 for seed 12), so the "closed" factors still carry a
little β in training and the model relies on it. **Inferred, not yet measured:**
these runs were not evaluated with soft gates, and no checkpoints were kept.
This is why 10 closures yield only 4 usable runs.

**4. Closure rate from the pre-repair start.** Seeds 11–30 at γ = 0.1: 10/20
close with the right structure [0.30, 0.70]. Seeds 1–10 under the same start,
mostly at γ = 0.03, closed on 2 of 10 seeds (1 and 7) [0.06, 0.51], counting
closure regardless of accuracy. The intervals overlap; the difference could be
γ, the seed sample, or chance, and is not claimed.

### Hard gates late in training, 6 seeds (3 October)

`results/gate_hard_6seeds.json`. Seeds 12, 13, 15, 25, 29, 30 — all of which
closed on 2 October, five of them losing accuracy at evaluation. Same
configuration as the 2 October no-wall arm, two arms per seed: soft gates
throughout, and hard (straight-through) gates from step 12 000. Every run is now
evaluated with both soft and hard gates. Step 1 matched 2 October in 12 of 12.

| seed | soft arm: cost, hard / soft token | hard-from-12 000 arm: cost, hard token | 2 Oct |
|---|---|---|---|
| 12 | 4.000 | 1.659 (corr 0.86), 0.9996 | 1.2985, 0.652 |
| 13 | **3.0995**, 0.9990 / 0.9997 | 3.0995, **0.9834** | 3.0995, 0.878 |
| 15 | **2.199**, 0.9928 / 1.0000 | 4.000 | 1.2985, 0.966 |
| 25 | 4.000 | 4.000, 0.9630 | 3.0995, 0.780 |
| 29 | 4.000 | 4.000 | 1.2985, 0.974 |
| 30 | 4.000 | **3.0995**, 0.9951 | 3.0995, 0.973 |

Separated (corr ≥ 0.99 and token ≥ 0.99): soft arm 2/6, hard arm 1/6.

1. **The planned test of the evaluation mismatch did not get its cases.** Only 4
   of 12 runs had closed by step 12 000 (a fifth, seed 13's hard arm, was closed
   by the switch itself), so the seeds did not repeat their 2 October behaviour;
   and
   neither closed run in the soft arm lost accuracy: hard vs soft token 0.9990 vs
   0.9997 and 0.9928 vs 1.0000. The gap is in the predicted direction and small.
   No collapsed run was available to explain.
2. **But the switch itself shows the model relies on partly-open factors.**
   Switching soft gates to hard at step 12 000 raised the task loss immediately:
   seed 13 from 0.004 to **0.937** — its soft order was 3.97 just before, and
   snapping to {0, 1} took transpositions' fourth factor away (order 3.00) —
   seed 12 from 0.005 to 0.123, seed 30 from 0.048 to 0.074. This is the soft/hard dependence measured
   in training rather than inferred from evaluation.
3. **Hard gates from step 12 000 are not a fix in this form.** The shock is
   recovered in 2 of 3 cases within 2 000 steps; seed 13 ended at 0.9834. Usable
   runs did not rise (1/6 vs 2/6). A switch earlier, an annealed switch, or hard
   gates throughout would avoid a late shock; none has been run.
4. **Saturation can dissolve** (seed 12, hard arm; see correction to point 3
   above), and seed 30 (soft arm) again shows the second lock: 23% unsaturated
   at step 4 000 and no closure.

Stopping rule status (Part 7): hard-gate training tried once, penalty from step 1
not yet. Best usable rate still about 4 in 20.

### Penalty from step 1, 3-seed probe (6 October)

`results/gate_early_probe_{A,B}.json`, driver `work/run_gate_early_penalty.sh`
with seeds 11–13. Both arms: γ = 0.1 ramped from step 1 over 4 000 steps (no
penalty-free warmup), first factor always on (`--min_order 1`; without it a
500-step smoke closed every factor by step 250 and the layer wrote nothing).
Decision rule written before the run: 0/3 usable stops an arm, 2–3/3 earns the
full 20 seeds.

| | seed 11 | seed 12 | seed 13 | usable |
|---|---|---|---|---|
| **A** pre-repair start, soft gates | 1.199, hard tok 0.022 | **1.2985**, hard tok 0.219, *soft tok 0.995* | 1.0995, hard tok 0.040 | **0/3** |
| **B** open start, hard gates from step 1 | **1.2985**, 1.0000 | **1.2985**, 0.9992 | **1.2985**, 1.0000 | **3/3** |

(cost in factors per token from the learned gate, hard evaluation; usable = class
split corr ≥ 0.99 and hard token accuracy ≥ 0.99.)

1. **Arm B reached the exact allocation in 3 of 3 runs** — one factor to every
   transposition, four to every 5-cycle, no under-allocation, token accuracy
   ≥ 0.9992 — with no supervision on the allocation. Before this, the exact
   allocation with accuracy kept had appeared in 1 of 20 runs (2 October) and in
   2 of 12 runs, both on one seed, in the earliest configuration. Its gate never saturated (≤ 2.4% of
   probabilities), and in seed 11 the allocation settled at 1.30 by step 750,
   together with the task.
2. **But it is still a race, not yet a reliable mechanism.** The 2 500-step smoke
   test of the *same configuration and seed* (seed 11) had not escaped: loss 2.30
   at step 2 500 with half the 5-cycles under-allocated, where this run was at
   0.004. bf16 run-to-run variation decides it, as on the plateau cells (Part
   3c). Three of three is [0.44, 1.00]; the rule needs 15 of 20.
3. **Arm A fails both ways.** Seeds 11 and 13 under-allocated 5-cycles (3 and 2
   factors) and never learned the task. Seed 12 found the exact allocation and
   scored **0.995 with soft gates but 0.219 with hard gates** — the first direct
   measurement of the soft/hard mismatch inferred on 2 October (Part 3e, the
   logit wall, point 3), and an extreme case of it.
4. Read together: training on the gates evaluation uses (hard, straight-through)
   from the first step removes the mismatch, and starting open with the penalty
   from step 1 avoids the warmup lock. Which of B's three differences from A
   matters (open start, hard gates, or both) is not separated.

Costs are required work per token; Path A still computes every factor, so none
of this is a wall-clock saving.

### Arm B on 20 seeds (7 October) — the stopping-rule run

`results/gate_early_B20.json`: open start, hard straight-through gates and the
penalty from step 1, first factor always on, γ = 0.1, seeds 11–30.

**13 of 20 usable** [0.43, 0.82] — and all 13 are *exact*: one factor for every
transposition, four for every 5-cycle, token accuracy 0.9986–1.0000. Costs,
sorted: 1.0 ×3, 1.0995 ×3, 1.2654, 1.2985 ×13.

1. **Every failure is the same failure.** All 7 under-allocate the 5-cycles
   (three runs gave them 1 factor, three gave 2, one 3.67 on average) and never
   learn the task (final loss 0.81–3.49). None of the earlier failure
   modes appears: nothing saturated open, and no run that found the allocation
   lost accuracy at evaluation (hard and soft evaluation agree in all 20).
2. **The outcome is visible by step 500, before the task is learned.** Order at
   step 500: 1.28–1.31 in 11 of the 13 successes (already the right allocation,
   at task loss ≈ 4), 1.00–1.17 in all 7 failures (one success, seed 18, was
   also at 1.17). Successes reach task loss
   < 0.1 by step 750–1 250, except seed 17 (step 18 000).
3. **Seed does not decide it.** Seed 11 succeeded in the 6 October probe and
   failed here; seeds 12 and 13 succeeded both times.
4. **Stopping rule (Part 7): triggered.** 13/20 is below the pre-registered
   15/20. By the rule as written, the per-token gate goes on hold and the
   per-layer learned order becomes the main deliverable. Any decision to amend
   the rule instead must be recorded here, dated, with its reason, before
   further runs.

**Decision, 8 October: the rule is applied, not amended.** No further runs aimed at
per-token reliability on this cell; 13/20 exact stands as the per-token result.
An amendment (one more attempt with automatic restarts of runs that are visibly
failing at step 500) was considered and declined: it would make the *recipe*
reliable rather than the mechanism, and it would spend more effort on the toy
cell while the two largest gaps — no measured speedup, no real task — stayed
untouched. The restart recipe is postponed, not discarded; it costs one night
whenever a write-up needs a reliability figure. Part 7 records the new plan.

### The A₅ experiment, first attempt (8–9 October)

The working recipe from S₅ (open start, hard straight-through gates and the
penalty from step 1, first factor always on, γ = 0.1), on the all-5-cycle
alphabet (p = 1.0), scored with `--demand model` (every token: D = 2).

**3-seed probe, `results/gate_a5_probe.json`: 0 of 3 usable** [0.00, 0.56].
All three runs fell from 4 factors to exactly 1.000 by step 250 and stayed there
for 20 000 steps; token accuracy 0.0663–0.0669 under both hard and soft gates.
One factor cannot represent A₅ (D = 2), so nothing was learned. Neither the
prediction (2 factors) nor its pre-registered failure (3 or 4) was tested: the
gate shut before the task existed.

**Diagnosis, 5 000-step smoke tests, seeds 11–12**
(`results/gate_a5_smoke_gamma.json`, `results/gate_a5_smoke_slowgate.json`):

| arm | step 250 | step 2 000 | final order | token (hard) |
|---|---|---|---|---|
| γ = 0 (no penalty) | 4.000 | 4.000, loss 1.3–1.5 | 4.000, 4.000 | 0.9986, 0.9993 |
| γ = 0.1 | 1.000 | 1.000 | 1.000, 1.000 | 0.0616, 0.0608 |
| γ = 0.1, gate lr 1e-4 (10× slower) | 1.21–1.25 | 1.000 | 1.000, 1.000 | 0.0608, 0.0634 |

1. **The penalty causes the collapse; the task gradient does not.** Without it
   the gate stays at 4 and the task is learned by step ~2 000–2 500.
2. **It is a race, and the penalty wins it at any penalty strength.** The gate
   shut at γ_eff = 0.006. Adam normalises step size, so a small but consistent
   push moves the gate at full speed while the task gradient is still noise.
   Slowing the gate 10× only delays the collapse from step 250 to 500; the task
   needs ~2 000 steps at full order.
3. **Why S₅ did not show this:** 90% of S₅ tokens are transpositions, learnable
   at 1 factor, so the task starts improving immediately and the 5-cycles push
   their factors back open. With every token a 5-cycle, 1 factor makes no
   progress on anything and nothing pushes back. The recipe was tuned on a mix
   where the cheapest order already learns something; that is our setup, not
   the mechanism.
4. **New hazard for the obvious fix.** In the γ = 0 runs the halting logits rose
   to 18.3–19.2 once the task was learned, above the float32 saturation point
   (17.33): 3 of 4 factors at exactly λ = 1 (sat 0.74–0.75), zero gradient. A
   plain penalty delay would therefore lock the gate open — the same lock as the
   September warmup, now in float32. The logit wall (Part 3e) exists for this.

**Delayed penalty, 10 000-step smoke (9 October)** — penalty off for 3 000
steps, ramped to γ = 0.1 by step 7 000 (`results/gate_a5_smoke_delay.json`,
`results/gate_a5_smoke_wallctl.json`):

| arm | task learned (loss < 0.05) | logits at end | final order | token hard / soft |
|---|---|---|---|---|
| delay, no wall, seed 11 | step ~2 500 | zmax 18.7, sat 0.74 | 4.000 | 0.9985 / 0.9985 |
| delay, wall 5, seed 11 | step ~2 000 | zmax 5.0, sat 0 | 4.000 at step 7 750 (crashed) | — |
| wall 5, no penalty, seeds 11, 12 | step ~2 000 / ~4 000 | zmax 5.0, sat 0 | 4.000, 4.000 | 0.9998 / 0.9343, 0.9960 / 0.7528 |

1. **The float32 lock is real.** Without the wall, the logits passed 17.33 by
   step 1 500 and three of four factors were frozen open before the penalty
   started. The wall prevents it (zmax held at 5.0, sat 0 throughout).
2. **The wall alone closes nothing** (control: order 4.000 in both runs), so any
   closure in the wall arm is the penalty's.
3. **Not yet known whether the penalty can trim a learned model.** The wall arm
   ran only 750 steps at full penalty before a CUDA "unknown error" stopped
   the job at step 7 750 (most likely a WSL/driver fault from the laptop
   sleeping, not a code error — the next job in the chain ran normally). Order
   was still 4.000, but the gate weight norm had started rising (4.03 → 4.81
   over the last 500 steps), so something was moving. One arm, unfinished:
   no conclusion. Seed 12 of the delay arms never ran.
4. **Soft evaluation is unreliable with the wall** (0.75–0.93 soft vs ≥ 0.996
   hard): the wall holds λ near 0.993, and the product over factors is visibly
   below 1. Models are trained and scored with hard gates, so this is a
   reporting caveat, not a defect.

**Why this matters for the main deliverable.** A layer built at the gate's
*largest* order saves nothing on the S₅ mix: the 5-cycles there genuinely need
4. A per-layer saving exists only where the hand-chosen order over-provisions —
which is exactly the A₅ case (transposition length says 4; the demand model says
2; fixed n_h = 3 is already known to solve it at 0.9994). So the A₅ run is not a
side test: on the group tasks it is the only route to a measured saving, and
Boxes is the other.

**Next:** the delayed-penalty, wall-5 arm at full length (20 000 steps),
γ ∈ {0.1, 0.3}, three runs each. Reading, fixed before the run:
- settles at **2** with token ≥ 0.99 → the prediction holds; go to 20 runs.
- settles at **3** with token ≥ 0.99 → the pre-registered failure of the
  prediction for learned models (the known risk: fixed n_h = 2 learned A₅ to
  only 0.1207 in 20 000 steps). Still a 25% smaller layer to build and time.
- stays at **4** → a trained model defends its factors; the penalty only acts
  before learning. Next lever: a larger γ after the task is learned.
- drops below 2 and loses accuracy → γ too strong once the task is learned.

**Result, 20 000 steps, wall 5, penalty from step 3 000 (9 October;
`results/gate_a5_delay_wall.json`)** — token accuracy with hard gates:

| γ | run 11 | run 12 | run 13 |
|---|---|---|---|
| 0.1 | 4.000, 0.9996 | 4.000, 0.9997 | **3.373**, 0.9997 (still falling: 3.58 at 9k, 3.38 at 20k) |
| 0.3 | **1.000, 0.0559** (shut 4 → 1 between steps 7k and 8k; never recovered) | 4.000, 0.9984 | **3.000**, 0.9994 (exactly 3 from step 12k on) |

1. **No run reached 2.** By the rule written before the run, the prediction
   fails for this recipe: five of six runs ended at 3–4 with accuracy ≥ 0.99.
   Per the plan, there is no 20-run follow-up of this recipe.
2. **The meaning attached to that failure condition is wrong for the run at 3.**
   The rule said a gate at 3 or 4 means the model "is not using the 3-D
   representation". But the irreducible representations of A₅ have dimensions
   1, 3, 3, 4, 5, and a 5-cycle has rank(ρ − I) = 2, 2, 4, 4 in the four
   non-trivial ones. A₅ is simple, so any non-trivial representation is
   faithful, and the only faithful representations with 5-cycle rank ≤ 3 are a
   single 3-dimensional irrep plus trivial parts. So a 3-factor model that
   tracks A₅ exactly and homomorphically *must* be using a 3-D representation —
   and is carrying one factor it does not need. That inference assumes exact,
   homomorphic per-head tracking, which is not verified (no checkpoint was
   saved). The rule stands as written; only its interpretation is corrected.
3. **What the gate does in practice: it trims greedily and stops one factor
   short.** Removing a factor whose β is not already ≈ 0 costs task loss under
   hard gates (run 13 at γ = 0.3: task loss 0.61 at step 7 000 as it dropped a
   factor, recovered by 8 000). Nothing drives the spare factor's β to zero first,
   so the last redundant factor is defended. A larger γ does not fix this — it
   turns the trim into a collapse (run 11).
4. **Outcomes are bistable here too** (4, 3.37, 3, 1 at similar settings), so
   these are counts, not trends.
5. **Reporting defect found:** the summary's "separated" criterion uses the
   correlation between allocation and demand, which is undefined when every
   token has the same demand (reported as 0.00). For single-demand alphabets
   only the settled order and accuracy are meaningful.

**For the deliverable:** one run in six gives a gate-chosen layer of 3 factors
at 0.9994, where transposition length says 4. Fixed n_h = 3 is known to solve
this task (0.9994, one run, §5.1). That is enough to build and time 3 against 4
— the first measurable saving — but the gate's choice is not yet reliable.

**Context for the 2-factor goal:** fixed n_h = 2 on this alphabet escapes the
plateau in 3 of 10 runs within 20 000 steps (`A5_c5:2`, Part 3b corrections),
so 2 factors are learnable, slowly and rarely. The demand model's capacity claim
is supported by those fixed-order runs; what failed is the gate's ability to
find that solution by trimming a trained model.

---

## Part 3f — State tracking in natural language: the Boxes dataset (21 September)

First experiment outside group word problems, run by Shashwat Chaturvedi on the
`nlp_experiment` branch (`experiments/hybrid_delta_attn.ipynb`; numbers
transcribed into `results/boxes_hybrid.json`).

**Task.** Kim & Schuster (ACL 2023) Boxes: a description of the contents of
seven boxes, then operations (move / remove / put), then a query about one box;
the model must write that box's contents. Scored by exact match on the answer.
Training uses the 78 402 examples with at most two operations; testing uses
2 000 examples for each of 0–4 operations and 1 020 with five or more, so three
or more operations is extrapolation.

**Models.** Three parameter-matched models (~469k), 2 layers, width 128, 4 heads
× 32, three seeds each: DeltaProduct at **fixed n_h = 2**, causal attention with
RoPE, and a hybrid that runs both branches and mixes their normalised outputs
with a learned weight. Separate pure-PyTorch implementation in float32, verified
against a sequential float64 reference; **no halting gate**.

| operations | DeltaProduct | attention | hybrid |
|---|---|---|---|
| 0 | 0.988 (0.981–0.993) | 0.979 (0.947–0.984) | 0.995 (0.993–0.998) |
| 1 | **0.968** (0.948–0.975) | 0.312 (0.243–0.387) | 0.933 (0.910–0.942) |
| 2 | **0.929** (0.872–0.936) | 0.240 (0.215–0.267) | 0.861 (0.818–0.863) |
| 3 | **0.886** (0.781–0.905) | 0.193 (0.156–0.238) | 0.774 (0.652–0.803) |
| 4 | **0.863** (0.742–0.887) | 0.243 (0.147–0.244) | 0.749 (0.692–0.758) |
| ≥5 | **0.818** (0.712–0.839) | 0.196 (0.111–0.225) | 0.649 (0.569–0.703) |

Median of three seeds, range in brackets.

**What this establishes.** The recurrent layer applies state updates in text that
an equal-size attention model does not: both read the description correctly at
zero operations, but from one operation onwards DeltaProduct holds at 0.968 while
attention falls to 0.312. It also extrapolates, reaching 0.818 at five or more
operations after training on at most two.

**What it does not establish.**

- **Nothing about adaptive order.** n_h is fixed at 2 and there is no gate. The
  decisive follow-up is an n_h ∈ {1, 2, 3} sweep: if n_h = 1 already matches,
  adaptive order has nothing to offer on Boxes.
- **Nothing general about attention.** This is one small configuration trained
  from scratch for 8 000 steps.
- **The hybrid comparison is confounded.** Parameter matching left the hybrid a
  feed-forward width of 256 against 455 for the DeltaProduct arm, and its learned
  mixing weight stayed near one half (0.51–0.59). On a pure-recall control task
  all three models tie, so neither branch holds an advantage that mixing could
  combine; a router between attention and recurrence is therefore **not**
  motivated by this data.

---

## Part 4 — Claim structure

**Claim 1 — capability at matched compute (measured, 3 of 5 points).**
Fixed order is integer-valued, so it occupies discrete points on the cost axis.
Where D varies across the alphabet, a fixed-order model must provision D_max
while an adaptive one pays E[D]. For every budget B with `E[D] ≤ B < D_max`, no
fixed-order model both fits the budget and solves the task.

*Status:* holds at p = 0.05, 0.1, 0.25. Oracle solves at cost 1.150 / 1.300 /
1.752 where the cheapest solving fixed order costs 4. The gap shrinks
monotonically as predicted by E[ℓ] = 1+3p (3.48× → 3.08× → 2.29×).
*Caveats:* the cost is **notional** — Path A still expands to 4·T and zeroes
betas, so wall-clock is unchanged (761s vs 756s at p=0.1). Realizing it needs
ragged expansion. And p=0.5 is inconclusive (Part 6.2).

**Claim 2 — the demand model (proved for the algebra; prediction pending).**
Part 3. This is now the headline. It is a correction to an assumption the field
shares, it is proved to machine precision for the algebraic half, and it makes
a sharp falsifiable prediction we have not yet run.

**Claim 3 — learnability (reframed 13 September).**
At p=1.0, D=2, so `n_h=2` is sufficient *in principle* — and `fixed2` scored
**0.1207**. Representable but not learned: the gap Shakerinava et al. (ICLR 2026)
describe for diagonal models, now shown for Householder products.

*Correction to the strength of this.* The 0.1207 is **one run of a bistable
cell**, and every such cell we have since measured escapes 10–33% of the time
(Part 3c). So the honest claim is not "representable but not learnable" — we have
a counterexample to that, `S5_c5_t:4` at 0.9994 — but the weaker and more precise:

> **Representable, and learnable only with low probability per run.** The
> optimisation reaches the solution on a minority of runs; the rest sit on a
> plateau at chance for the whole budget.

That is still the Shakerinava phenomenon, and arguably a sharper version of it,
because it is a statement about the *optimisation landscape* rather than about
what the architecture can express. It also makes the claim measurable — an escape
rate — rather than a binary. Restating it needs escape fractions on the p=1.0
cell, which we have not run.

---

## Part 5 — The full experimental record

### 5.1 Matched-compute sweep (20 000 steps, 1 layer, 12 heads × 32, seed 666)

Alphabet is 10 transpositions + 24 five-cycles throughout; only the mixture rate
p changes. **The generated group is S₅ for every p < 1.**

| p | E[ℓ] | fixed1 | fixed2 | fixed3 | fixed4 | oracle | verdict |
|---|---|---|---|---|---|---|---|
| 0.05 | 1.150 | 0.2009 | 0.6148 | 0.8689 | 0.9999 | **1.0000** @ 1.150 | CLAIM HOLDS |
| 0.10 | 1.300 | 0.1583 | 0.4659 | 0.9696 | 0.9999 | **0.9999** @ 1.300 | CLAIM HOLDS |
| 0.25 | 1.752 | 0.0671 | 0.1176 | 0.5516 | 0.9997 | **0.9999** @ 1.752 | CLAIM HOLDS |
| 0.50 | 2.501 | 0.0402 | 0.0313 | 0.0332 | 0.7070 | 0.9875 @ 2.501 | INCONCLUSIVE |
| 1.00 | 4.000 | 0.0648 | 0.1207 | **0.9994** | 0.9995 | 0.9998 @ 4.000 | DOES NOT HOLD ✓ |

(token accuracy; the p=1.0 verdict is the intended negative control and it
fired correctly.)

> **Caveat, 13 September.** Every cell above is a **single draw**, and cells of
> this kind are bistable (Part 3c). Values near 0.999 or near 0.04 are deep in a
> basin and robust; the mid-range entries — `fixed2` 0.4659, `fixed3` 0.5516,
> `fixed4` 0.7070, `fixed3` 0.8689 — are exactly where a bistable cell lands when
> sampled once and must not be read as capability measurements. The sweep needs
> re-running over runs before any of these numbers goes in a document.

### 5.2 Error forensics at p = 0.1 (20 000 steps)

| | fixed1 | fixed3 | fixed4 |
|---|---|---|---|
| token / seq | 0.1822 / 0.0000 | 0.9740 / 0.7145 | 0.9999 / 0.9940 |
| per-hard-token success q | 0.7124 | 0.9947 | 1.0000 |
| continue/start ratio | 6.5× | 38.2× | 937.6× |
| mean wrong-run length | 30.13 | **2.00** (median 1) | 1.11 (max 2) |
| accuracy vs target ℓ, spread | 0.0859 | 0.0046 | 0.0002 |
| sequences with any error | 2000/2000 | 571/2000 | 12/2000 |

Readings:

- **Not a decoding failure.** F4 is flat for every arm — accuracy does not
  depend on which element the cumulative target is. The readout is fine; the
  recurrence is where it breaks.
- **Errors are clustered, not independent and not absorbing.** Both naive models
  were falsified before the run: independent errors predict seq acc
  `0.974^128 = 0.034` (observed 0.7145); permanent corruption predicts token acc
  0.746 (observed 0.9740).
- **`fixed3` recovers.** 38× persistence but a *median wrong-run of 1* and mean
  2.0. A corrupted group-tracking state has no mechanism to heal itself, so
  something outside the transition matrices is carrying information — the
  additive `v kᵀ` channel is the obvious suspect and is unexamined.
- **`fixed1` is the control that shows what real failure looks like**: q collapses
  to 0.71, accuracy hits 0.02 by the third hard token, wrong-runs average 30.

### 5.3 Noise floor — important

`fixed3` at p=0.1, **identical config and seed**, produced **0.9696 / 0.6645**
in the sweep and **0.9740 / 0.7145** in the forensics run. bf16 kernels use
non-deterministic atomics. So:

> **run-to-run noise is ≈0.005 token accuracy and ≈0.05 sequence accuracy.**
> Any claim resting on a smaller difference is noise. Every headline number
> from here needs ≥3 seeds with a reported spread.

The big contrasts (0.55 vs 0.9994) are two orders of magnitude outside this.
The small ones are not.

> **Scope correction, 13 September.** The figure above was measured on `fixed3`
> at p=0.1 — a cell that *reliably solves*. It does not transfer, and applying it
> to bistable cells is what licensed every single-seed claim in this document.
>
> On a **bistable** cell the spread at fixed seed is **0.84 token accuracy**:
> `A5_c5:n_h=2` produced 0.1177 and 0.9598 from the identical configuration and
> the identical seed, with byte-identical step-1 loss. There the run either
> escapes the plateau or does not, and bf16 non-determinism alone decides.
>
> So there are two regimes and they need different statistics:
>
> | cell type | statistic to report |
> |---|---|
> | reliably solving | accuracy ± spread over ≥3 runs; ±0.005 is the floor |
> | bistable | **escape fraction k/N plus the sorted list.** Never a mean |
>
> Three seeds is not enough on a bistable cell. Telling a 25% escape rate from a
> 75% one at p<0.05 with 80% power needs about **20 runs per arm** (`AUDIT.md` §5).

---

## Part 6 — Open problems and what could kill us

### 6.1 `fixed3` at p ≤ 0.25 is unexplained

At p=0.1 the alphabet generates S₅, so D(5-cycle)=4 and `fixed3` provably
cannot represent one. Yet it reaches q=0.9947 per hard token.

The multi-head escape is closed: a direct sum of representations is faithful iff
some summand is, and every non-faithful rep of S₅ has kernel ⊇ A₅ (the only
normal subgroups are 1, A₅, S₅). A faithful summand needs rank 4; each head has
rank ≤ 3. So **no arrangement of 12 heads can track S₅ exactly by homomorphic
per-head representations.** Whatever `fixed3` is doing is non-homomorphic or
approximate.

**Decisive cheap test:** train `fixed3` at length 128, evaluate at 512. An
approximation degrades with length; an exact mechanism does not. We already have
this machinery.

### 6.2 p = 0.5 is an optimization cliff, not a capability failure

The alphabet — hence D — is identical at p = 0.25 and p = 0.5. Yet `fixed4`
solved p=0.25 by step 2000 and was still at 0.7070 after 20 000 steps at p=0.5.
The loss traces show delayed onset: `fixed4` sat at 4.67 until step 11 000, and
the oracle sat at ~3.94 until step 16 000 then fell to 0.07 by 20 000. Both were
still descending when the budget ran out.

So difficulty is **non-monotone in the mixture rate**, peaking near p=0.5, with
identical capability requirements at either side. Hypothesis: at low p the
transposition-dominated data provides a curriculum that bootstraps the harder
tokens; at p=1 there is only one demand to learn; at p=0.5 neither dominates.
Testable by curriculum (train at 0.1, fine-tune at 0.5). Worth a paragraph in
the paper either way.

### 6.3 The costs are notional

Path A gives no wall-clock saving. This is the single largest gap between what
we claim and what we have demonstrated, and a reviewer will raise it first. We
should raise it first instead.

### 6.4 The gate is trained, but it closes only sometimes

`budget_loss` is now wired into a training loop and run (Part 3d). The gate
closed on **1 of 3 seeds**; on that seed it reproduced the oracle allocation
exactly, and on the other two it never left full order.

Everything in Parts 3, 4 and 5 still comes from the **oracle**, which reads
ground truth, and is labelled as such. Exactly one result in this document —
Part 3d — describes a learned adaptive model, and it is a 1-in-3 outcome, not a
working mechanism. The phrase "trained end-to-end" is accurate for that single
run and must not be attached to any other table here. (Overclaiming this on a
departmental form is the error recorded in Part 9; the safeguard is that the
claim now has a specific run, a specific seed count, and a specific file behind
it.)

### 6.5 Risks that would genuinely hurt

| risk | severity | mitigation |
|---|---|---|
| Group closure prediction fails | high — Claim 2 dies | it is one run; find out now |
| Learned gate never matches the oracle | high | measure the oracle–gate gap early |
| Ragged expansion is slower in practice | medium | benchmark before promising speedups |
| Everything is S₅-specific | medium | the demand model is general; show it on S₄, A₄, S₃ |

---

## Part 7 — Plan forward

### Review, 11 October — before the next run

**A measurement we already had.** Every matched-compute sweep and escape-census
run recorded its training time (`train_s`), and fixed orders build a layer with
exactly that many factors. Same script, same GPU, 20 000 steps, batch 128:

| order | 20k-step training time, s (5 sweeps, §5.1 files) | relative to 4 |
|---|---|---|
| 1 | 342.5 – 379.6 | 0.45 |
| 2 | 440.7 – 454.9 | 0.58 |
| 3 | 598.7 – 606.3 | 0.79 |
| 4 | 758.7 – 761.3 | 1.00 |

So the cost of each fixed order **is measured**, five times over, within 3%
at orders 2–4 (11% at order 1). The status line "no wall-clock measured" was true only of
the *adaptive* layer (Path A computes every factor: oracle 756 s vs fixed4
761 s). The fixed-order run planned for tonight would have re-measured this.
These are end-to-end training times; inference has not been timed.

**What the record supports, by route.**

- *Per-token gate on the heterogeneous S₅ mix:* works in 13 of 20 runs, exact
  allocation at 1.2985 factors per token. Notional only, because Path A runs
  every factor. If cost tracks expanded sequence length — the table above
  suggests it roughly does — a ragged expansion at 1.30 would land between
  fixed1 and fixed2, about half of fixed4. **That is a projection, not a
  measurement.**
- *Per-layer order built at the gate's choice (the 8 October deliverable):*
  on the S₅ mix the largest order any token needs is 4, so the saving is zero —
  Tier 3's own abandon condition (Part 7b) holds there. On A₅ the gate chose 3
  in 1 of 6 runs (and 2 factors were never reached), and a 3-factor layer
  trains in 0.79 of the time. A plain sweep over n_h = 1–4 finds 3 just as
  well, so on present evidence the gate adds nothing over a sweep there.
- *A₅ learnability:* with 4 factors and the gate open, the task was learned in
  12 of 12 runs; fixed 3: 1 run (0.9994); fixed 2: 3 of 10 (Part 3b census).

**Ragged expansion is smaller than the plan assumed.** In this layer a factor
whose hard gate is 0 has β = 0, an exact identity in the kernel
(`work/check_beta_identity.py`). The first factor is always on, so the decay
term (placed on the first sub-step) is always kept. Dropping closed sub-steps,
moving the query to the last kept sub-step, and packing sequences with
`cu_seqlens` (already accepted by `chunk_gated_delta_rule`) should give the same
output as Path A, which can be checked by test. With packed sequences the cost
is the sum of kept factors, not the maximum per position over the batch, so the
"batching dilutes the saving" argument in Part 7b (granularity constraint) may
not apply. Unverified until built and timed.

**Amendment, 11 October (decided before any code or run).** The main
deliverable changes from "a layer built at the gate's largest order" to
**"the per-token gate with ragged expansion at inference, with a measured
speedup at equal accuracy"**.

*Reason.* The per-layer route met its own abandon condition (Part 7b, Tier 3:
"learned order gives no accuracy or cost advantage over the best hand-chosen
n_h") on the S₅ mix, where the largest order needed is 4; on A₅ the gate does
no better than a sweep over n_h. The per-token gate is the part that works
(13/20 exact), and ragged expansion is what turns its allocation into time.
*What does not change.* The stopping rule's verdict stands: 13/20 is the
reported per-token reliability, and no further runs aim at raising it on the
S₅ cell except the restart rule (step 3 below), which is reported as a training
procedure with its cost. Per-layer results already obtained stay in the record.

**Plan under the amendment:**
1. *Ragged expansion* — built 11 October, inference only:
   `src/ragged_expansion.py` (packing, float64 self-test R1–R11 with four
   mutants), the layer switch `ragged` in `gated_deltaproduct.py`
   (PATCH(ragged-expansion); previous file kept as `.orig.ragged`), and
   `work/check_ragged.py` (real-kernel equality with Path A, mutants, timing
   grid). Pass criteria and the speedup reading are written in that file's
   header before its first run: speedup claimed only where ragged at the
   learned S₅ allocation is ≥ 10% faster than fixed n_h = 4.
2. *Accuracy through Path B on a trained model* — needs a checkpoint (none has
   ever been saved): one arm-B run with saving, then evaluate with ragged on.
3. *Restart rule* — failure is visible by step 500; restart failing runs and
   report reliability per attempt and per procedure.
4. *Real task* — Boxes order sweep (n_h ∈ {1, 2, 3}); if order matters, the gate
   on Boxes; otherwise chess.
5. *A₅* — written up as it stands; no further gate tuning now.

**Ragged expansion, first measurement (11 October; `results/ragged_bench_v1.json`).**
RTX 4050 laptop, 1-layer model at the experiment size, forward pass only,
median of 30 after warm-up; synthetic allocation matching the learned one
(order 4 for 10% of tokens, else 1; mean 1.23–1.30); random weights; peak
memory ≤ 3.3 GB, so no spill.

*Correctness (real kernel):* all pass. Ragged vs Path A relative L2 7.6e-3; both
judged against a float64 reference of the recurrence, ragged 7.05e-3 and Path A
6.57e-3 — the same bf16 noise. All gates open: identical (0.0). Three mutants
detected (0.26–0.78). An earlier version of the check failed on an
argmax-agreement criterion (0.9907) that was unsuitable for random-weight
logits; it was replaced by the stricter reference comparison, not loosened
(header of `work/check_ragged.py`).

| batch | length | fixed 1 | fixed 4 | Path A | ragged | ragged / fixed 4 |
|---|---|---|---|---|---|---|
| 1 | 128–2048 | 3.1 | 4.7 | 4.9–5.0 | 7.1–7.3 | 1.49–1.56 |
| 8 | 128, 512 | 3.2 | 4.7, 5.7 | 4.9, 5.8 | 7.3, 7.8 | 1.55, 1.37 |
| 8 | 2048 | 4.7 | 16.0 | 16.1 | 12.1 | **0.76** |
| 32 | 128 | 3.1 | 5.4 | 5.6 | 13.7 | 2.56 |
| 32 | 512 | 4.6 | 15.6 | 15.8 | 12.4 | **0.79** |
| 32 | 2048 | 18.3 | 62.2 | 62.9 | 42.9 | **0.69** |

(ms per forward pass)

1. **By the rule fixed before the run, a speedup is claimed in 3 of 9 cells** —
   the three large ones: 0.69–0.79 of the fixed-4 time. This is the first
   measured wall-clock saving from the adaptive layer. It is inference, one
   GPU, one small model, a synthetic allocation, and one run of the grid.
2. **Small sizes are slower** (1.4–2.6×). There the GPU is mostly idle: fixed 1
   and fixed 4 differ by only 1.6 ms, and ragged adds a roughly fixed ~2.3 ms
   of packing and launch overhead. Batch 1 — the case Part 7b's target
   statement named — shows no saving at any length tested.
3. **The large-size saving is well short of what the arithmetic allows.** At
   batch 32, length 2048, fixed orders cost about 15 ms per factor
   (18.3 / 35.7 / 48.3 / 62.2), so 1.30 factors per token should land near
   23 ms; ragged took 42.9. Ragged at order 4 everywhere took 78.3 against
   Path A's 62.9: about 15 ms of packing overhead. Version 1 packed by stacking
   all four factors and selecting with boolean masks (several full copies and a
   host synchronisation per mask).
4. **One cell is unexplained:** batch 32, length 128, ragged took 13.7 ms at
   the mixed allocation but 9.7 ms with every factor kept. Not investigated;
   a second run of the grid will show whether it repeats.

**Version 2 of the packing** (same day): index_copy per factor into
preallocated buffers, one host synchronisation, no stacked copy. Same CPU
self-test (R1–R11, all pass). To be re-checked and re-timed with the same
command; the version-1 numbers above stay as the record.

**Version 2, two runs of the grid (12 October; `results/ragged_bench_v2a.json`,
`_v2b.json`).** Correctness unchanged (same C1–C4 values). Ragged / fixed 4,
run a / run b:

| batch × length | v1 | v2a | v2b |
|---|---|---|---|
| 8 × 2048 | 0.76 | 0.67 | 0.69 |
| 32 × 512 | 0.79 | 0.71 | 0.70 |
| 32 × 2048 | 0.69 | 0.67 | 0.65 |
| small cells (6) | 1.37–2.56 | 1.35–1.62 | 1.38–1.63 |

1. **The claim holds in the same 3 of 9 cells, now 0.65–0.71 of the fixed-4
   time**, and the two runs agree to within 0.02. Fixed-order times also repeat
   within ~3%.
2. **My prediction was wrong.** I expected the new packing to bring the large
   cells to 0.4–0.5; it moved them by 0.02–0.09. With every factor kept, ragged
   is still 12–14 ms slower than Path A at 32 × 2048 (74.0–75.9 vs 62.1–62.7),
   so most of the overhead is not where I assumed. Not guessing again:
   `work/profile_ragged.py` splits each forward into GPU kernel time by
   category before the next change.
3. The version-1 anomaly at 32 × 128 (ragged slower at the mixed allocation
   than at all-4) did not repeat (7.7–7.8 vs 9.0 ms).
4. Small sizes stay ~2.5 ms slower than Path A in both runs: a fixed cost.

**Where the time goes (12 October; `results/ragged_profile.json`,
`work/profile_ragged.py`).** GPU kernel time per forward, ms, batch 32 × 2048:

| | wall | recurrence | packing | matmul | conv | copies/elementwise | GPU idle | launches |
|---|---|---|---|---|---|---|---|---|
| fixed 4 | 60.9 | 35.4 | 0.3 | 6.0 | 5.5 | 11.7 | 0.6 | 62 |
| ragged, learned allocation | 41.0 | 13.2 | 6.8 | 6.0 | 5.5 | 2.7 | 5.4 | 163 |
| ragged, all factors kept | 74.6 | 40.9 | 6.9 | 6.0 | 5.5 | 4.8 | 8.8 | 163 |

1. **The recurrence does scale with the kept factors:** 13.2 against 35.4 ms
   (0.37; the allocation alone predicts 0.33). That part of the mechanism works.
2. **The rest of the gap is overhead, in three places.** Packing 6.8 ms (one
   index_copy kernel alone is 5.6 ms, because version 2 writes every factor of
   every token, the closed ones into scratch rows); GPU idle 5.4 ms (host
   synchronisation and 163 kernel launches against 62); and per-factor
   projections and convolutions, 11.5 ms, which are computed for all four
   factors regardless.
3. **The fixed-4 baseline itself carries ~9 ms of copies** — upstream's
   interleaving stacks four tensors to build the expanded sequence. Part of
   ragged's measured advantage is therefore just not paying that copy, which a
   better-written fixed layer would not pay either. **The 0.65–0.71 figure
   overstates what adaptivity buys against a well-written fixed layer**, and is
   not to be quoted without this caveat.
4. Running the packed sequences through the kernel costs ~15% more than the
   batched layout for the same work (40.9 vs 35.4 ms with every factor kept).

**Fused factors (built 12 October).** The four factors' projections run as one
matmul, and their convolutions as one depthwise convolution, on concatenated
weights. The result is already interleaved, so Path A needs no copy and ragged
expansion needs one gather of the kept rows only. It is offered to the fixed
layer as well, and **the claim rule was revised before its first run**:
fused ragged *plus the learned gate's own cost* (previously left out, because
the timed runs used external gates) against the *faster* of the per-factor and
fused 4-factor baselines, at least 10% faster. Correctness checks C5–C8 and
CPU self-tests R12–R14 added; R12–R14 pass.

**Fused factors, two runs (12 October; `results/ragged_bench_v3a.json`,
`_v3b.json`).** Correctness: all checks pass; fused and per-factor paths are
bitwise identical (C5, C8: 0.0), fused ragged matches the float64 reference as
closely as Path A (7.05e-3 vs 6.57e-3). Times in ms, run a / run b:

| batch × length | fixed 4 (original) | fixed 4 (fused) | fused ragged | gate | (ragged + gate) / best 4 |
|---|---|---|---|---|---|
| 32 × 2048 | 63.8 / 63.3 | 56.2 / 56.2 | 39.0 / 39.6 | 3.1 / 3.2 | **0.75 / 0.76** |
| 32 × 512 | 14.9 / 16.4 | 14.1 / 13.9 | 10.3 / 10.0 | 0.8 / 0.7 | **0.79 / 0.77** |
| 8 × 2048 | 17.0 / 16.2 | 14.0 / 14.1 | 12.9 / 9.7 | 0.7 / 0.7 | 0.98 / 0.74 |
| 6 small cells | | | | | 1.19–2.68 |

1. **The fair figure: 0.75–0.79 of the best 4-factor time, gate cost included,
   at batch 32 (lengths 512 and 2048), in both runs** — about 21–25% faster.
   This replaces the 0.65–0.71 figure, which compared against the original
   code's copy-heavy layer and left out the gate.
2. **Batch 8 × 2048 is not claimed:** it met the rule in one run (0.74) and not
   the other (0.98). Small sizes remain slower and very noisy run to run.
3. Removing the copies made the fixed layer 7.6 ms faster (63.8 → 56.2) but the
   ragged path only ~2.5 ms faster (41.8 → 39.0), so most of the earlier
   advantage over the original code was indeed the copies.
4. **Overhead is still large.** Fixed fused orders 1–4 at 32 × 2048 cost 20.0 /
   31.6 / 44.5 / 56.2 ms, so 1.30 factors per token would be ~23.5 ms with no
   overhead; fused ragged takes 39.0. With every factor kept it takes 80.4 against
   56.2, about 24 ms of packing and kernel overhead at full length. The learned
   gate itself costs 3.1 ms (about 5% of the baseline).
5. ~~Not yet measured: accuracy of a trained model through this path.~~
   Measured 13 October, below.

**Trained models through the ragged path (12–13 October;
`results/gate_B_saved.json`, `results/ragged_trained.json`,
`results/ragged_trained_check2.json`; `work/check_ragged_trained.py`).** Best
recipe, seeds 11–13, models saved with `--save_dir`: seeds 12 and 13 exact
(1.2985, token 0.9998 and 1.0000), seed 11 failed (1.199, token 0.42) — 2/3.

| model | Path A token / seq | ragged token / seq | prediction agreement | error vs float64 ref, ragged / Path A | margin at disagreements (median, all tokens) |
|---|---|---|---|---|---|
| seed 12 (trained) | 0.9998 / 0.9795 | 0.9996 / 0.9595 | 0.99951 | 1.88e-2 / 1.81e-2 | 2.31 (7.81) |
| seed 13 (trained) | 1.0000 / 0.9965 | 0.9999 / 0.9940 | 0.99997 | 2.32e-2 / 2.14e-2 | 0.56 (8.08) |
| seed 11 (not learned) | 0.4218 / 0 | 0.4210 / 0 | 0.90032 | 2.65e-2 / 2.07e-2 | 0.0625 (0.44) |

Speed with the trained gate inside the forward pass, ragged / best 4-factor
layer: 32 × 512: 0.80, 0.80; 32 × 2048: 0.73, 0.74 (seeds 12, 13).

1. **On both trained models the ragged path passes every check**: reload (T0),
   same token accuracy within 0.0002 and ≥ 99.95% identical predictions (T1),
   and its recurrence output is as close to a float64 reference as Path A's
   (T3). The speedup with the real gate matches the synthetic figure.
2. **Sequence accuracy is a little lower on the ragged path** (0.9795 → 0.9595
   and 0.9965 → 0.9940): a handful of borderline tokens flip, and one flip
   spoils a sequence. Token accuracy is lower in all three models (by 0.00002, 0.0002 and 0.0008).
   Likely cause, not tested: the model was trained under Path A's exact bf16
   rounding, and any other rounding is a small perturbation of it. To be stated
   with the result, not hidden.
3. **T1 failed as written on the untrained model (seed 11): 90% agreement.**
   Recorded as failed. The checks added before the re-run show why: its
   ragged output is as close to the float64 reference as Path A's (T3 pass),
   and the disagreeing tokens sit at near-ties (median margin 0.0625, one bf16
   step at that scale, against 0.44 overall). A model that has not learned the
   task has near-flat scores, and rounding decides between them. This is the
   second time an argmax-agreement criterion misled us on a non-confident
   model; such criteria are only used on trained models from now on.
4. **T2 was a test-design error, corrected 13 Oct**: the broken variant changes
   only multi-factor tokens (~10%), so its agreement sits near 0.90 regardless;
   T2 now asks whether T1 catches it, which it does on all three models.

### Boxes order sweep — pre-registered 13 October, before any run

*Question.* Does the number of factors per token matter on real text? If it
does not, a learned per-token order has nothing to save on Boxes and the next
real task is chess board state.

*Setup.* `work/exp_boxes_order.py`, using `work/boxes_lib.py`: a verbatim copy
of the code cells of Shashwat's notebook (`experiments/hybrid_delta_attn.ipynb`,
branch `nlp_experiment`, commit abd1349), whose own self-test still passes.
Pure-PyTorch float32 DeltaProduct (no decay gate, β ∈ (0, 2)); his settings: 2
layers, d 128, 4 heads × 32, 8 000 steps, batch 64, train on ≤ 2 operations,
exact match on 0…8 (≥ 3 = extrapolation). DeltaProduct only, `n_h` ∈ {1, 2, 3},
5 runs each (seeds 0–4, interleaved). **Same parameter count at every order**
(469,120, his `n_h = 2` model): the MLP width is 589 / 455 / 321, so more
factors never means more parameters. Each order has its own storage key (the
notebook's key has no `n_h` and would overwrite one order with the next).
Self-test W1–W6, every check confirmed to fail on a broken version (key
without `n_h`, no size matching, wrong buckets in E, no count criterion).

*Decision rule.* E(run) = unweighted mean exact match over ops = 3, 4, ≥ 5;
m1 = median E at `n_h = 1`; for k ∈ {2, 3}: gain_k = median E(k) − m1,
above_k = runs at k with E > m1.
- **ORDER MATTERS**: some k has gain_k ≥ 0.05 and above_k ≥ 80% of its runs →
  the learned gate goes on Boxes next.
- **NO GAIN**: every k has gain_k < 0.03 → one factor is enough here; chess next.
- **UNINFORMATIVE**: no run of any order reaches E = 0.10 → fix the setup.
- **INCONCLUSIVE**: anything else → 5 more runs per order (seeds 5–9), once,
  same rule on all 10; still INCONCLUSIVE counts as NO GAIN.

*Amendment, 13 October, after the smoke run and before the sweep*
(`results/boxes_order_smoke.json`, 500 steps, one run per order). The smoke
run confirmed the pipeline (vocabulary 123, 78,402 training examples, no
unknown words, all three orders at 469,120 parameters) and the time per order
(0.5 / 1.15 / 2.3 min per 500 steps for 1 / 2 / 3 factors). At 500 steps E was
0.32 / 0.07 / 0.01: higher orders learn more slowly at the start. One run each,
at 6% of training: not evidence about the final answer, but it means a fixed
8 000-step budget could rank the orders by speed rather than by capability.
Changes: **16 000 steps** (not 8 000); quick evaluation on 500 examples per
bucket every 2 000 steps; and a convergence condition, checked after
UNINFORMATIVE and before the rest of the rule: **NOT CONVERGED** if for any order
the median quick-eval E rose by more than 0.03 over the last quarter of
training (step 12 000 → 16 000) → no decision, rerun at twice the steps.
Self-test W5b, confirmed to fail without the override and with the rise
measured from the start. Expected cost ≈ 2.1 h per seed, ≈ 10.5 h in all.

*Prediction (weakly held).* NO GAIN is plausible: a "move" needs about two
state changes, but the sentence gives the model several tokens to make them.
Shashwat's three `n_h = 2` runs (E ≈ 0.88, 0.86, 0.75) show the run-to-run
spread this rule has to see through.

**Result (13–14 October; `results/boxes_order.json`, `.log`): ORDER MATTERS.**
15 of 15 runs completed (16 000 steps, none diverged), 469,120 parameters each.
E = mean exact match on 3, 4 and ≥ 5 operations, sorted per run:

| factors | E per run, sorted | median E | ops = 1 / 2 (median) | minutes per run |
|---|---|---|---|---|
| 1 | 0.595 0.721 0.810 0.842 0.888 | 0.810 | 0.952 / 0.890 | 14 |
| 2 | 0.833 0.844 0.915 0.916 0.924 | 0.915 | 0.971 / 0.945 | 33 |
| 3 | 0.810 0.884 0.889 0.909 0.909 | 0.889 | 0.978 / 0.944 | 75 |

Rule: 2 factors gain +0.105 with 5/5 runs above the 1-factor median (0.810);
3 factors gain +0.079 with 4/5 (the fifth, 0.8096, sits just below 0.8103).
Convergence: rise over the last quarter −0.005 / +0.029 / −0.005 — none above
0.03, so the verdict stands; the 2-factor curve is the one still edging up,
which could only widen its lead over 1 factor, whose curve is flat.

1. **On real text, at equal parameter count, two factors per token beat one.**
   Every 2-factor run (lowest 0.833) is above the 1-factor median; the
   1-factor runs are spread widely (0.595–0.888), so its best run matches
   the 2-factor runs but its typical run does not.
2. **Three factors add nothing over two** here (median 0.889 vs 0.915, ranges
   overlapping), with the MLP at 321 instead of 455. The useful order on
   this task and size is 2.
3. The gain is largest where the state changes most (≥ 5 operations: 0.761 →
   0.873) and present in distribution too (2 operations: 0.890 → 0.945).
4. Training cost in this pure-PyTorch implementation: 14 / 33 / 75 min per
   run. Not an inference measurement and not the fla kernel.
5. Scope: one task, one model size, 5 runs per order, float32 pure-PyTorch
   DeltaProduct without the decay gate. Shashwat's 8 000-step 2-factor runs
   (E 0.745–0.877) were below these, so the longer budget mattered.

Per the rule, **the learned gate goes on Boxes next.**

### Restart rule — pre-registered 14 October, before any run

*Question.* The best recipe reaches the exact allocation in 13 of 20 runs, and
every failure is visible at step 500. Does restarting the visibly failing runs
turn it into a training procedure that reliably delivers the allocation? This
measures a *procedure*, not the mechanism: the 13/20 stays the per-run figure.

*Rule.* A learned run whose mean order (training batch, hard gates) at step 500
is below **1.20** is stopped and started again from seed base + 1000 per
attempt; at most **5 attempts**, the fifth runs to the end without the check,
so every base seed ends in a trained model. Threshold from the 20-run study
(`results/gate_early_B20.json`): at step 500 the 7 failures sat at 1.00–1.17,
the successes at 1.28–1.31 (one at 2.56) except seed 18 at 1.167. So 1.20
restarts all 7 failures and 1 of 13 successes there — chosen on those runs,
so the test is on new ones.

*Run.* `work/run_gate_restart.sh`: the best recipe unchanged (open start, hard
straight-through gates and the penalty from step 1, ramp 4 000, γ 0.1, first
factor always on, 20 000 steps), **new base seeds 31–50**, models saved. Code:
`--restart_below / --restart_at / --max_attempts` in `exp_gate_train.py`;
self-tests T22–T22g (the threshold on the 20-run study, seeds per attempt,
the unchecked last attempt, the summary), each confirmed to fail on a broken
version (check at every step after 500, threshold flipped or at 1.25, the
same seed every attempt, last attempt still checked, missed failures counted
wrongly). About 6 h.

*Reported.* Exact k/20 per base seed with a 95% interval; restarts; first
attempts that passed the check (expected about 12/20 from the study); runs that
passed the check and still failed ("missed"); seeds where every attempt was
below the threshold; restarted steps against completed steps.

*Decision.* **≥ 18/20 exact**: the restart procedure becomes the standard way
to train the gate from here on (Boxes included), always reported as a
procedure with its restart count, never as the gate's per-run reliability.
**< 18/20**: the step-500 signal is not enough; 13/20 remains the figure and
gate runs on Boxes are made without restarts, with enough runs to report a
fraction.

**Result (14 October; `results/gate_restart.json`, `.log`; 6.0 h): 18 of 20
exact** [0.70, 0.97] — the rule's bar, met exactly. **Applied: the restart
procedure is the standard way to train the gate from here on**, reported as
a procedure with its restart count.

| | count |
|---|---|
| base seeds ending exact | 18 / 20 |
| first attempts that passed the check | 11 / 20 (expected about 12) |
| restarts | 24 (12 000 steps against 400 000 completed: 3% extra) |
| attempts per seed | 1 ×10, 2 ×2, 3 ×4, 5 ×4 |
| passed the check and still failed ("missed") | 1 (seed 36, attempt 3) |
| all four checked attempts below 1.20 | 4 seeds (33, 39, 42, 44); the unchecked 5th attempt succeeded in 3 |

1. **Attempts that passed the check succeeded 15 of 16 times**; counting the
   two unchecked fifth attempts that were also above 1.20, 17 of 18. The one
   miss (attempt 3 of seed 36) sat just above the bar (1.211) and failed in a
   new way: it learned the task only partly (token 0.848, loss stuck near 0.4–0.6)
   with 3.75 factors on 5-cycles and 1.10 on transpositions — not the
   under-allocate-and-learn-nothing failure seen before.
2. **The check also restarts runs that would have succeeded.** Seed 39's fifth
   attempt was at 1.129 at step 500 (below the bar), ran unchecked, and reached
   the exact allocation — as seed 18 did in the 20-run study (1.167). Seed 44's
   fifth attempt (1.147) failed. How many of the 24 restarted attempts would
   have succeeded is unknown: they were stopped.
3. **Checked attempts passed 16 of 40 times** [0.26, 0.55] — 11/20 first attempts,
   5/20 later ones. Small numbers; no trend is read from the split.
4. **Against the plain recipe** (13/20, `gate_early_B20`), 18/20 is higher but not
   separated by these sample sizes (Fisher exact, two-sided p = 0.13; intervals
   [0.43, 0.82] and [0.70, 0.97] overlap). The claim is "the procedure reached
   18/20 at 3% extra compute", not "restarts are proven better".

### Learned gate on Boxes — pre-registered 22 October, before any run

*Question.* On Boxes, can the learned gate keep the accuracy of 2 factors per
token while using fewer, and does it give the second factor to the tokens that
change the state? The gate gets no labels; token kinds are used only to read
its allocation afterwards.

*Setup.* `work/exp_boxes_gate.py`. Identical to the order sweep (Shashwat's
model and data, 2 layers, 16 000 steps, batch 64, same optimiser, schedule,
seed scheme and quick evaluation) except: maximum order **K = 2** (the sweep
found a third factor adds nothing); the gate is `src/gating.py`'s HaltingGate
with the S₅ recipe (open start, bias 4; hard straight-through gates from step 1;
first factor always on; penalty γ × mean order over non-padding tokens,
ramped linearly from step 1 to γ over 4 000 steps). Model size held at
469,120 parameters: the gate's 16.8k parameters come out of the MLP width
(455 → 422). Float32 throughout. Self-tests G1–G7: an open gate reproduces
the original branch bit for bit; a closed second factor equals a
one-factor recurrence (float64, 4e-16); per-token closing equals an explicit
loop; the training loop reproduces `run_one`'s losses exactly; padding is
excluded from the order; the γ choice and decisions on known answers. Each
new check confirmed to fail on a broken version (gate on v instead of β,
rule giving query tokens 2, padding counted, first factor allowed to close,
γ chosen by best dev score alone).

*Arms.*
- **Reference, not re-run:** fixed 1 (median E 0.810) and fixed 2 (0.915),
  the 5 + 5 runs of `results/boxes_order.json`, same code path.
- **Hand rule:** the second factor only for tokens inside the operation
  sentences (boxes_lib's 'operation' kind), one elsewhere; 469,120
  parameters; seeds 0–4. Tells whether an uneven allocation is enough here.
- **Learned gate, probe:** γ ∈ {0.01, 0.03, 0.1}, seeds 0–1. γ is chosen on
  the **dev split**, never the test split: the largest γ whose median dev E
  is within 0.02 of the best γ's median dev E.
- **Learned gate, final:** 5 new runs (seeds 10–14) at the chosen γ, reported
  on the test split.

*Restarts are not used,* although the restart procedure was adopted on
14 October: its threshold (1.20 at step 500) is the S₅ allocation's, and Boxes
has no known target allocation, so the rule cannot be applied as written.
Every run logs its order and loss at step 500, so a Boxes signal can be
defined afterwards if outcomes split.

*Decision (final stage, 5 runs).* Bar E ≥ 0.885 (fixed-2 median − 0.03).
- **NOT CONVERGED** (checked first, as in the sweep): no decision; rerun longer.
- **GATE WORKS:** median E ≥ 0.885 and median order ≤ 1.5 → the gate on Boxes
  goes into the faster path next (timing).
- **ACCURACY, NO SAVING:** median E ≥ 0.885, order > 1.5 → reported; a larger γ
  is tried only if the hand rule shows a cheaper allocation works.
- **LOSES ACCURACY:** median E < 0.885 → read against the hand rule: if the
  rule reaches the bar, the structure exists and the gate misses it (look
  for our setup's fault first); if not, the second factor's value is not
  concentrated in operation tokens.
- **Mechanism, reported either way:** operation tokens get a higher mean
  order than description tokens in at least 4 of 5 runs.
- **Hand rule:** median E ≥ 0.885 → an uneven allocation is enough here.

*Prediction (weakly held).* The hand rule comes close to fixed 2 at about
1.1–1.3 factors per token; the gate settles between 1.1 and 1.5 at γ 0.01–0.03
and collapses towards 1 at γ 0.1.

*Cost.* Probe: 11 runs × ~33–35 min ≈ 6.3 h (one night). Final: 5 runs ≈ 3 h.

*Amendment, 22 October, after the 300-step smoke test and before any full run*
(`results/boxes_gate_smoke.json`; no accuracy is readable at 300 steps — every
E ≤ 0.12). **The order bar is raised from 1.5 to 1.6.** Reason: the hand rule's
order is fixed by the data, not by training, and on the test split it is
**1.508** — the test split spreads examples evenly over 0–8 operations, so
operation tokens are about half of all tokens. The 1.5 bar was set expecting
1.1–1.3 without measuring this; it would have called the reference allocation
itself "no saving". 1.6 = at least 40% of second factors skipped. Our error,
corrected before results. Also added: the order on test examples with 0–2
operations (like training data) is reported beside the overall order, and
every trained model is saved (`checkpoints/boxes_gate/`) for timing later.
Self-tests updated (bar boundary; per-bucket order; save and reload).
Smoke observations, not evidence: in 300 steps the gate had already dropped
to 1.71–1.92 at every γ (penalty still ≤ 0.0075 then), and gave operation
tokens fewer factors than description tokens. Adam moves the gate at full
speed under any consistent push, which is what made the A₅ gate collapse;
whether the task pushes back here is what the probe measures.

*Probe result and an added read-out, 22 October, after the probe and before the
final stage* (`results/boxes_gate.json`, `results/boxes_gate_probe.log`).
- **Hand rule (5 runs; this arm is complete):** E [0.621 0.696 0.793 0.796 0.859],
  median **0.793** at 1.508 factors per token. Below the bar, and not above
  fixed 1 (0.810): giving the second factor to the operation sentences does not
  recover fixed 2's accuracy.
- **Learned gate, probe (2 runs per γ; not a decision):** γ 0.01: E 0.909 / 0.689
  at 1.31; γ 0.03: 0.898 / 0.657 at 1.09 / 1.13; γ 0.1: 0.926 / 0.888 at 1.089
  in both. γ chosen on dev by the pre-registered rule: **0.1**.
- **Why three runs report the identical order 1.089.** Read from the saved models
  (40 test examples per bucket; a diagnostic, not a new run). In layer 1 the gate
  closes the second factor on every token. In layer 0 it opens it on exactly the
  100 item words (the words that follow "the": glass, fig, ...) at every
  occurrence, and on nothing else: 17.6% of tokens in one of two layers gives
  1.088. Layer 0's gate reads only the token embedding, so an allocation by token
  type is all it can express there. Open logits are 0.1–4.6 (still trainable);
  closed logits are −6 to −70 (gradient effectively zero: a closed factor cannot
  reopen). The pre-registered mechanism check (operation > description) is
  therefore expected to fail — item words occur in descriptions as well as
  operations. The check stays as written and is reported.
- **Added read-out for the final stage; prediction written before it runs.** In
  at least 4 of 5 final runs: (a) layer 1 gives the second factor to fewer than
  1% of tokens, and (b) at least 90% of layer 0's second factors fall on item
  words. If both hold, the next step is to rebuild layer 1 with one factor (a
  saving with existing kernels, timed) and to send layer 0 through the faster
  path. No decision rule is changed.

### Learned gate on Boxes — result, 23 October (final stage, applied as pre-registered)

`results/boxes_gate.json`, `results/boxes_gate_final.log`. γ 0.1 (chosen on dev),
seeds 10–14, 16 000 steps.
- **Test E** (sorted): [0.720 0.769 0.835 0.855 0.856], median **0.835**; order
  [1.029 1.029 1.032 1.082 1.088], median **1.032**. Late rise −0.018 (converged).
- **Decision: LOSES ACCURACY** (bar 0.885). Read against the hand rule, which also
  missed the bar (0.793): by the pre-registered reading, the second factor's
  value on Boxes is not concentrated in the operation sentences.
- **Mechanism check:** operation > description in 1 of 5 runs → no.
- **Added read-out (prediction of 22 Oct):** (a) layer 1 (the second layer) gives the second factor
  to < 1% of tokens — **held, 5/5** (0.0000 in every run). (b) ≥ 90% of layer 0's (the first
  layer's) second factors on item words — **failed, 2/5** (needed 4/5). Seeds 11 and 13
  keep the item words (93 and 99 of 100); seed 10 keeps only "."; seeds 12 and
  14 keep only "contains" and "is". In those three the remaining open logits are
  0.06–0.3, on the edge of closing. Across all seven γ-0.1 runs (2 probe +
  5 final) layer 1 is fully closed in 7/7, and the item-word allocation appears
  in 4/7 (E 0.926, 0.888, 0.856, 0.769).
- **What the numbers say, narrowly:** at γ 0.1 the gate ends at about one factor
  per token, and its accuracy is that of fixed 1 (median 0.810), not fixed 2
  (0.915). The order at step 500 (1.44–1.64) does not separate outcomes.
- **Our setup first.** The gate's order keeps falling late in training (e.g.
  seed 12: 1.096 → 1.029 after step 12 000) and rises by at most 0.002 between
  evaluations: closed logits sit at −6 to −70 in the models inspected, where the gradient is effectively zero, so a factor closed by the
  penalty cannot be reopened by the task. This is the same one-way pattern as the
  A₅ collapse, and it comes from our training choices (an unbounded push under
  Adam), not from DeltaProduct. Layer 0's gate also reads only the word, so it can
  allocate by word type and nothing finer. Both are fixable; next experiment to be
  pre-registered before it runs.

### Learned gate on Boxes — follow-up, pre-registered 23 October, before any run

*Correction to the diagnosis above, first.* The result entry blamed mainly the
one-way closing (closed logits at −6 to −70). Checking the order sweep's training
losses points to a larger cause, also ours: **the penalty was mis-scaled for this
task.** On the training data (0–2 operations) fixed 1 and fixed 2 end with task
losses of 0.0042 and 0.0021 (medians, `results/boxes_order.json`) — a gap of about
0.002 — while closing every second factor saves γ itself, 0.01 to 0.1, i.e. 5 to
50 times the gap. The training objective therefore prefers one factor, and the
gate did what it was asked; the second factor's value shows mostly on 3+
operations, which training never sees. The γ grid was copied from S₅, where
under-allocation makes the task unlearnable, without measuring this gap. The
logit floor (the S₅ wall, which on 2 October removed saturation but did not
change closure) is held back as the fallback: it can slow the closing but cannot
change which allocation the objective prefers. One change at a time.

*Arms* (`work/exp_boxes_gate.py --stage layers21 | probe2`; driver
`work/run_boxes_followup.sh`; same data, model, optimiser, 16 000 steps).
- **layers21 — fixed orders [2, 1]:** 2 factors in the first layer, 1 in the
  second (the second layer really built with one factor's projections, not
  multiplied by zero), no gate, MLP width 522 → 469,120 parameters. Seeds 0–4.
  Tests the gate's one consistent finding (second layer closed in 7/7 runs).
- **probe2 — the learned gate at γ 0.001 and 0.003** (below and above the 0.002
  gap), seeds 0–2, otherwise the 22 October recipe.

*Decisions, written before the runs.*
- **layers21** (5 runs): median E ≥ 0.885 → **[2,1] HOLDS**: time [2, 1] against
  fixed 2 (training step and inference, same GPU) next — the first measured
  saving on a real task, with the order taken from the gate. Below → **[2,1]
  LOSES ACCURACY**: the second layer's second factor matters, and the gate closed
  it because of the penalty, not because it was unneeded.
- **probe2:** γ chosen on dev among {0.001, 0.003} by the existing rule (largest γ
  whose median dev E is within 0.02 of the best). The final stage (5 runs, seeds
  10–14, decided by the 22 October rule: E ≥ 0.885 and order ≤ 1.6) runs only if
  the chosen γ's median dev E is at least 0.855 (bar − 0.03); below that, the
  probe already rules it out and the logit floor is next.
- **Read-out:** median change in order over the last quarter of training;
  "settles" = falls by at most 0.02.

*Prediction (weakly held).* At γ 0.001 the order ends at 1.5 or above and
settles; at γ 0.003 it ends between 1.1 and 1.5 and settles. **The scale
explanation fails** if at γ 0.001 the order still ends below 1.2 or is still
falling (late drop > 0.02): then the one-way closing is the cause and the logit
floor is next. No prediction for [2, 1].

*Self-tests* G8–G9 added (layer orders as built, the second layer's projections
really one factor's, order 1.5 reported, a one-factor branch with open gates
equals the plain branch exactly, 469,120 parameters, save and reload, decisions
and read-outs on known answers); G1–G7 unchanged and passing. Each new check
confirmed to fail on a broken version (layers built [1, 2]; second layer not
rebuilt; bar exclusive; late change from the first evaluation; probe read with
2 seeds; earlier γs allowed in the follow-up choice; quick-eval orders not
recorded; settle threshold strict): 8/8 caught.

*Cost.* Smoke (300 steps, both stages) ~20 min; layers21 ~2 h; probe2 ~3.4 h;
one night. The 22 October decision (LOSES ACCURACY at γ 0.1) stands.

*Amendment to the follow-up, 23 October, after a review and before any follow-up
run.* Reason: Boxes scores are too noisy for the bars we set. With a constant
learning rate, E swings by about 0.03 between evaluations within one run (e.g.
fixed 2, seed 2, steps 8k–16k: 0.845, 0.803, 0.902, 0.893, 0.826), so final-step E
depends on where training stops. The **late average** — E averaged over the last 4
quick evaluations (steps 10k–16k, already logged, 500 examples per bucket) — is
steadier. Recomputed from existing logs (post hoc; no decision is changed by it):

| arm | median final E | median late average |
|---|---|---|
| fixed 1 | 0.810 | 0.799 |
| fixed 2 | 0.915 | 0.872 |
| fixed 3 | 0.889 | 0.897 |
| gate γ 0.1, final | 0.835 | 0.842 |
| hand rule | 0.793 | 0.785 |

The 0.885 bar was fixed 2's final-E median − 0.03, and that median was the lucky end
of its runs: resampling fixed 2's own runs, a 5-run median reaches 0.885 only 68% of
the time on final E and 32% on the late average. Changes, all before running:
- **All follow-up decisions use late averages.** Bar = median late average of the
  fixed-2 reference − 0.03 (0.842 with the 5 sweep runs).
- **layers21: 8 runs** of [2, 1] (seeds 0–7), **plus 3 fixed-2 runs** in this code
  (seeds 5–7; self-test G10 shows the build is identical to the sweep's fixed 2), so
  the reference has 8. [2,1] HOLDS if its median late average ≥ the bar. The exact
  Mann–Whitney p against fixed 1 and fixed 2 is reported beside it, not used.
- **probe2:** γ chosen on the dev late average (dev now evaluated at the same 4
  late points); the final stage runs only if that median ≥ bar − 0.03, and is
  decided with late averages (median ≥ bar and order ≤ 1.6 → GATE WORKS; ≥ bar and
  order > 1.6 → ACCURACY, NO SAVING; else LOSES ACCURACY).
- **The γ-scale explanation is a hypothesis, not a finding.** Against it: at γ 0.1
  the gate kept the item words in 4 of 7 runs with logits still trainable (locally
  the task valued them above the penalty), and γ 0.01 was not more accurate than
  γ 0.1. The 0.002 gap is between separately trained models, not the value of a
  factor inside the gated model. The prediction and its failure condition stand.

*Corrections to earlier entries* (kept as written above; corrected here):
- The 23 October result said the γ-0.1 gate's accuracy "is that of fixed 1". On
  late averages it is **below fixed 2** (one-sided Mann–Whitney p = 0.014) and **not
  distinguishable from fixed 1** (two-sided p = 0.69). The LOSES ACCURACY decision
  stands, more firmly.
- The order sweep's ORDER MATTERS rests on 5 runs per arm: fixed 2 above fixed 1 at
  one-sided p = 0.028 (two-sided 0.056), on final E and on late averages alike.
  Real but modest evidence.
- The restart rule's 18/20 is not significantly better than 13/20 without restarts
  (Fisher p = 0.13), and is on S₅ only.

*Self-tests* G9–G10 extended (late average, bar, decisions, Mann–Whitney exact p
against a known value, references, the fixed-2 build identical to the sweep's, dev
evaluated only at the 4 late points). 17 deliberately broken versions, all caught.
*Cost* ≈ 8.5 h (layers21 ≈ 5 h, probe2 ≈ 3.6 h).

### Learned gate on Boxes — follow-up — result, 24 October (applied as pre-registered)

`results/boxes_gate.json`, `results/boxes_followup_{layers21,probe2,driver}.log`.
Scores are late averages (mean E over the last 4 quick evaluations).

**[2, 1] fixed orders — [2,1] HOLDS.**
- [2, 1] (8 runs): [0.716 0.851 0.876 0.877 0.877 0.879 0.881 0.908], median **0.877**.
- Fixed 2 (8 runs: 5 sweep + 3 new): [0.675 0.792 0.836 0.853 0.856 0.872 0.895
  0.902], median **0.855** → bar 0.825. The three new fixed-2 runs were weak (0.792,
  0.675, 0.836), which lowered the bar; [2, 1] also passes the 5-run bar (0.842) and
  the old final-E bar (its final-E median is 0.898 ≥ 0.885), so the decision does not
  depend on that.
- Reported, not used: Mann–Whitney two-sided p = 0.33 against fixed 2, 0.13 against
  fixed 1 (one-sided [2,1] > fixed 1: 0.064). Post hoc, runs with a late average
  ≥ 0.85: fixed 1 1/5, fixed 2 5/8, [2, 1] 7/8.
- In-distribution (0–2 operations) exact match, 2 operations: [2, 1] 0.932–0.971,
  new fixed-2 runs 0.920–0.934.
- **Time, measured on whole runs** (same GPU, same 16 000 steps, including the quick
  evaluations): [2, 1] **23.6–23.7 min** per run (8 runs), fixed 2 **33.4–33.5 min**
  (3 runs) — 29% less. Not yet a controlled benchmark; that is the next step.
- **Caveat on the order sweep's ORDER MATTERS:** with the 3 new fixed-2 runs, fixed 2
  above fixed 1 is one-sided p = 0.11 (was 0.028 with 5 runs). The effect of the
  second factor on Boxes is real in direction but noisy.

**Gate at γ 0.001 / 0.003 (probe, 3 runs each).**
- γ 0.001: late E 0.871 / 0.784 / 0.890 at order 1.516 / 1.410 / 1.472; order change
  over the last quarter, median −0.033 → **still falling**.
- γ 0.003: late E 0.876 / 0.767 / 0.858 at order 1.432 / 1.339 / 1.404; median
  change −0.003 → **settles**.
- **Prediction:** γ 0.003 between 1.1 and 1.5 and settling — held. γ 0.001 at 1.5
  or above and settling — **failed** (1 of 3 at ≥ 1.5, still falling). By the
  pre-registered failure condition ("still falling at γ 0.001"), **the γ-scale
  explanation fails as stated.** Noted, not used to rescue it: at γ 0.001 the order
  moves both ways between evaluations (e.g. 1.614 → 1.438 → 1.540 → 1.472 → 1.409),
  unlike the monotone fall at γ 0.1.
- **γ chosen on the dev late average: 0.003** (median 0.855 ≥ bar − 0.03 = 0.795)
  → **the final stage runs** (seeds 10–14, decided with late averages: median
  ≥ 0.825 and order ≤ 1.6 → GATE WORKS).
- **Where the gate puts the factors** (read from the 6 saved models, 40 test examples
  per bucket; diagnostic): the second layer gives the second factor to 0.04–6.6% of
  tokens; the first layer to 67–100% of tokens, always including all 100 item words.
  At small γ the gate's own allocation is therefore approximately [2, 1] — in 6 of 6
  runs — the order the fixed [2, 1] model was built at (chosen from the γ-0.1 runs).

**What this is, narrowly:** on Boxes, a layer built at the order the gate chose —
two factors in the first layer, one in the second — matches fixed 2 under the
pre-registered rule and trained in 29% less time per run. The 8 October main
deliverable (order learned by the gate, layer built at that size, measured saving at
equal accuracy) has its first instance on a real task, pending a controlled timing
benchmark (training step and inference) and with the noise caveats above.

### Timing [2, 1] against fixed 2 — pre-registered 24 October, before any run

*Question.* The whole training runs of [2, 1] took 23.6 min against 33.4 min, but whole
runs include evaluation and logging. How much faster is the model itself, at training
and at prediction time?

*Setup.* `work/time_boxes_orders.py`. Models at 469,120 parameters: fixed 2 [2, 2]
(d_ff 455, trained checkpoint `fixed22_g0.0_seed5.pt`), [2, 1] (d_ff 522,
`fixed21_g0.0_seed0.pt`), fixed 1 [1, 1] (d_ff 589, untrained: the recurrence has no
data-dependent control flow, so weights do not change its cost), and the gated model
at γ 0.003 (`learned_g0.003_seed0.pt`) as a control. Workloads on real Boxes data, the
same batches for every model (fingerprinted): **infer64** — inference, 12 test batches
of 64, 2 per bucket; **infer1** — single-example latency, 24 test examples, 4 per
bucket; **train64** — one full training step (forward, loss, backward, clipping,
AdamW) on 8 training batches of 64. 3 warm-up passes, then 30 rounds with the models
in a seeded random order each round. Ratio per round = time([2, 1]) / time(fixed 2);
reported as the median with a bootstrap 95% interval of the median (2 000 resamples).
The GPU must be idle (the script refuses to start while a training script runs).

*Decision (per workload).* **MEASURED SAVING** if the interval lies entirely below
0.95 — then the median saving is stated; **NO CLAIM** otherwise (a difference of a
few percent can come from the harness). The gated control's ratio is reported: near 1
shows its per-token saving is notional without the faster path.

*Scope, stated in any claim.* The pure-PyTorch recurrence used in every Boxes run
(`boxes_lib.deltaproduct`, a Python loop over chunks of T × n_h micro-steps), float32,
RTX 4050 laptop GPU. Not the Triton kernels used on S₅; the ratio there may differ.

*Prediction (weakly held).* infer64 and train64: ratio 0.65–0.85 (the second layer's
recurrence halves; embeddings, MLPs and the head do not). infer1: no prediction — at
batch 1 the cost is dominated by launching operations, not arithmetic. Gated control
on infer64: 0.95–1.10.

*Self-tests* B1–B5: widths 455 / 522 / 589 and 469,120 parameters each (fixed 1's width
equals the sweep's); checkpoints loaded when present and a mismatched one refused;
the harness on sleeps of known length (4 ms vs 8 ms → ratio 0.5 and a claim; 8 vs 8 →
no claim); the round order varies and is reproducible; the interval is of the median
(not of single rounds); the 0.95 threshold; inference changes no weights, a training
step does. 9 deliberately broken versions, all caught. End-to-end smoke on CPU
(2 rounds, not evidence) ran every workload and loaded all three checkpoints (exact
match on the timed batches 0.884 / 0.947 / 0.941).

*Cost.* About 5–10 minutes of GPU time, after the γ-0.003 final stage finishes.

### Learned gate on Boxes, γ 0.003 final stage — result, 24 October (as pre-registered)

`results/boxes_gate_final_g0.003.log`, `results/boxes_gate.json`. Seeds 10–14,
late averages.
- Late E (sorted): [0.697 0.850 0.851 0.862 0.876], median **0.851** (bar 0.825);
  order [1.309 1.354 1.395 1.398 1.461], median **1.395** (≤ 1.6). Late rise +0.005
  (converged); order settles (median change −0.003).
- **Decision: GATE WORKS.** It also passes the 5-run bar (0.842). It would not pass
  the old final-E bar (final-E median 0.871 < 0.885), which was replaced before these
  runs because fixed 2 itself fails it in about 2 of 3 repeats.
- Reported, not used: Mann–Whitney two-sided p = 0.83 against fixed 2, 0.42 against
  fixed 1 — with 5 runs and this noise, not distinguishable from either.
- Mechanism check (operation > description): 1 of 5 → no, as expected from 22 Oct.
- **Notional, not measured:** the order 1.395 is factors per token with closed factors
  multiplied by zero. Measured below: the gated model is *slower* than fixed 2. The
  real saving comes from building the layer at the gate's order ([2, 1]).

### Timing [2, 1] against fixed 2 — result, 24 October (as pre-registered)

`results/boxes_timing.{log,json}`. RTX 4050 laptop GPU, float32, pure-PyTorch
recurrence, 30 rounds, GPU idle (checked by the script).

| workload | fixed 2 | [2, 1] | ratio [2, 1] / fixed 2 (95% interval) | decision |
|---|---|---|---|---|
| inference, batch 64 | 20.66 ms/batch | 16.44 | **0.793** (0.790–0.796) | MEASURED SAVING |
| inference, batch 1 | 8.18 ms/example | 7.16 | **0.895** (0.859–0.936) | MEASURED SAVING |
| training step, batch 64 | 118.64 ms/step | 86.83 | **0.725** (0.714–0.755) | MEASURED SAVING |

- Floor, fixed 1: 0.584 / 0.794 / 0.423. Control, the gated model: **1.046 / 1.090 /
  1.112** — 5–11% slower than fixed 2 (the gate's own cost, with no factor skipped).
- Predictions: batch-64 inference and training in 0.65–0.85 — held (0.793, 0.725);
  gated control 0.95–1.10 on batch-64 inference — held (1.046). Batch 1 had no
  prediction; the saving there is smaller (launch-bound).
- The training-step ratio (0.725) agrees with the whole-run ratio (23.6 / 33.4 = 0.71).

**What is now established on Boxes, narrowly.** A layer built at the order the gate
chose — two factors in the first layer, one in the second — keeps fixed 2's accuracy
under the pre-registered rule (late averages, 8 runs each) and is **21% faster at
prediction (batch 64), 10% at batch 1, and 27% faster per training step**, measured.
Scope: this implementation (pure PyTorch, float32) on one laptop GPU; the ratios with
the Triton kernels may differ. Accuracy comparisons rest on 5–8 noisy runs per arm.
This is the 8 October main deliverable's first instance on a real task.

### Sequence accuracy on the faster path — pre-registered 25 October, before any run

*Question* (queue item 2b). On trained S₅ models the faster path keeps token accuracy
(seed 12: 0.99979 → 0.99959) but whole-sequence accuracy drops (0.9795 → 0.9595). The
paths compute the same mathematics and differ only in bf16 rounding inside the kernel.
Is the drop (a) a flaw of the faster path — ours, to fix — or (b) sensitivity of the
model's sequence accuracy to any change of recurrence rounding, with Path A looking
better only because the model was trained through its rounding? Earlier evidence points
both ways: equal distance of the two paths from a float64 recurrence (0.0188 vs 0.0181),
but disagreeing tokens are not near-ties (median logit margin 2.3 vs 7.8 overall).

*Method.* `work/check_seq_accuracy.py`. A third path X: the same model and weights with
only the recurrence exact — fla's `chunk_gated_delta_rule` / `chunk_delta_rule` swapped,
for the call, for a float64 sequential recurrence with fla's convention, on the same bf16
inputs, output rounded once to bf16. Paths scored on each checkpoint's full test set:
A (Path A, per-factor, the training path), R (fused + ragged, the timed faster path), X.
Models: every saved S₅ checkpoint — `checkpoints/gate_B/*` (seeds 11–13) and
`checkpoints/gate_restart/*` (the 20 restart-procedure runs).

*Checks per model, on the GPU, before scoring.* G0 the checkpoint reloads (Path A token
accuracy within 0.002 of the training record); G1 the swap is active and exact (on 8 test
sequences, X's recurrence output within 5e-3 relative L2 of the validated float64
reference, and at most half of Path A's error); G2 G1 can fail (Path A's own error
exceeds 5e-3). A model failing a check is excluded from the decision and reported.

*Decision* (pooled over models that learned the task — Path A token accuracy ≥ 0.99 —
weighted by sequences):
- **NO GENERAL DROP:** seq(A) − seq(R) ≤ 0.005 — the seed-12 drop does not generalise;
  the per-model range is stated with the speed claim.
- **FASTER-PATH FAULT:** seq(X) ≥ seq(R) + 0.01 and |seq(X) − seq(A)| ≤ 0.005 — fix the
  faster path before the speed claim is used.
- **ROUNDING SENSITIVITY:** seq(X) ≤ seq(A) − 0.01 — exact arithmetic also loses
  sequences; the faster path is not at fault; the claim states how much sequence
  accuracy moves between numerically equivalent paths.
- **UNCLEAR:** anything else.
Reported either way: per-model accuracies, discordant sequence counts for A/R, A/X, R/X,
and Path A's logit margin where paths disagree.

*Prediction (weakly held).* ROUNDING SENSITIVITY: X lands near R, not near A.

*Self-tests* X1–X4 (CPU, float64): the exact recurrence equals the validated reference to
0.0 with and without decay; the drop-in kernel returns the input dtype, equals the exact
recurrence on the bf16 inputs rounded once, and refuses initial state, final state,
packing and the un-normalised convention; the swap is always undone, also after an
error; scoring, discordance, pooling and every decision branch on known answers. 10
deliberately broken versions, all caught. G1–G2 run on the GPU with the real model.

*Cost.* ~20–30 min of GPU time for 23 checkpoints (the exact path runs at batch 32).

### Sequence accuracy on the faster path — result, 25 October (applied as pre-registered)

`results/seq_accuracy.{log,json}`. 23 checkpoints; every per-model check passed (G0
reload, G1 swap exact: X's recurrence error 1.5–1.7e-3 against Path A's 1.0–8.4e-2;
G2). 20 learned the task (Path A token accuracy ≥ 0.99); seeds 11, 2036 and 4044 did not
and are reported only. 2 000 test sequences per model.

| pooled over 20 models | Path A (trained path) | R (faster path) | X (exact recurrence) |
|---|---|---|---|
| sequence accuracy | 0.9891 | 0.9706 | **0.9956** |

- The drop is general: pooled A − R = 0.0185; per model, sorted: −0.0010 +0.0000 +0.0005
  +0.0010 +0.0025 +0.0025 +0.0035 +0.0045 +0.0095 +0.0125 +0.0200 +0.0200 +0.0255 +0.0280
  +0.0295 +0.0300 +0.0365 +0.0380 +0.0415 +0.0655.
- **The exact recurrence beats both bf16 paths.** X > A in 19 of 20 models (pooled
  +0.0065); X ≥ R in 20 of 20 (pooled +0.025). Discordant sequences, summed over the 20
  models: A right / X wrong 35 vs X right / A wrong 294; R right / X wrong 37 vs X right /
  R wrong 1 036; A right / R wrong 953 vs R right / A wrong 213.
- Where A and X disagree the model was near a tie (median logit margin 0.0–0.9); where A
  and R disagree it was not (median 0.4–5.6, typically 2–4).
- **Decision: UNCLEAR.** FASTER-PATH FAULT needed |X − A| ≤ 0.005 (it is 0.0065, X above
  A); ROUNDING SENSITIVITY needed X ≤ A − 0.01 (X is above A). **The prediction (rounding
  sensitivity) failed:** exact arithmetic does not lose sequences — it gains them. The
  rule did not anticipate the exact path being better than the training path.
- **What the numbers say, narrowly (not a decision):** both bf16 paths lose sequences to
  rounding, and the faster path loses about four times as many as Path A (−0.025 vs
  −0.0065 against X). The faster path's mathematics is proven in float64; its larger loss
  is numerical, in the bf16 kernel on packed sequences — so it is ours to fix, not a
  property of the model. The earlier output-error check (8 sequences, first layer, mean
  relative error) was too coarse to see it.
- **Consequence for the S₅ speed claim:** "0.73–0.80 of the 4-factor time at equal token
  accuracy" stands; at the sequence level the faster path is currently 1.85 points worse
  (pooled; up to 6.6 on one model). The claim may not say "equal accuracy" without this.
- **Next, to be pre-registered before running:** localise where the faster path's extra
  error enters (per layer, per position, chunk boundaries of the packed sequences) and
  try a more accurate kernel for the packed recurrence (fla's fused-recurrent form keeps
  the state in float32), scored the same way against X.

### Fixing the faster path's sequence accuracy — pre-registered 25 October, before any run

*Question.* Why does the faster path (fused factors + ragged packing, bf16 chunk kernel)
lose ~4x more sequences to rounding than Path A, and does a different kernel fix it?

*Suspects on our side, each tested on the real trained models* (`work/check_ragged_fix.py`):
1. **Packing / indexing** — proven in float64 only on small CPU cases. Path R64 runs the
   real packed recurrence with an exact float64 kernel that honours the sequence
   boundaries; it must reproduce X (Path A, exact).
2. **Fused projections** (our concatenated short convolution) — path F (fused Path A) must
   reproduce Path A.
3. **Gate decisions flipping** between paths — a discrete change: fraction of tokens whose
   number of factors differs from Path A's, per layer, every path.
4. **Chunk density** — fla's chunk kernel solves a 64x64 triangular system per chunk and
   stores the inverse in bf16 (`solve_tril`, output dtype = input dtype). In Path A about
   two thirds of a chunk's micro-steps are closed factors (beta 0); after packing every
   micro-step is a real update, often near a reflection (beta ~ 2). Measured as the error
   of each path's recurrence output against a float64 recurrence on the same inputs, per
   layer and per quarter of the sequence.
*Candidate kernels for the packed recurrence:* **R16** — the same chunk kernel with
float16 inputs (10-bit mantissa instead of 7); **Rrec** — fla's
`fused_recurrent_gated_delta_rule` (sequential, state in float32, accepts packed
sequences). Paths scored on each learned model's full test set: A, F, X, R, R64, R16, Rrec.

*Checks per model:* G0 reload (Path A token accuracy within 0.002 of the record); G1 both
exact paths are exact (X's and R64's recurrence output within 5e-3 of float64 in every
layer and position block — without this, a swap that failed to apply would make R64 equal
R and read as a packing bug); G2 G1 can fail (Path A's error exceeds 5e-3 somewhere).

*Decision* (pooled over learned models passing G0–G2):
- **Step 1, our side, first.** PACKING BUG if agreement(R64, X) < 0.9999 or
  |seq(R64) − seq(X)| > 0.002; FUSION BUG if agreement(F, A) < 0.9999 or
  |seq(F) − seq(A)| > 0.002. Either → stop and fix our code; candidates are not judged.
- **Step 2.** A candidate is ACCURATE if seq ≥ seq(A) − 0.002 and token ≥ token(A) − 0.0001.
- **Step 3.** Timed on one model (seed 12) as in the 13 October claim (test sequences
  joined to 32 x 512 and 32 x 2048, learned gate inside the forward, median of 30 after 5
  warm-up): ADOPTED if ACCURATE and time / best fixed-4-factor time ≤ 0.90 at both sizes
  (the original "at least 10% faster" rule); if several, the fastest. None → the S₅ speed
  claim keeps its sequence-level caveat.

*Prediction (weakly held).* Step 1 clear; the faster path's extra error grows along the
sequence (chunk density); Rrec ACCURATE but slower than the bf16 chunk kernel, R16 in
between. Whether either is adopted: no prediction.

*Self-tests* P1–P4 (CPU, float64): the packed exact recurrence equals the validated
reference with sequence boundaries (0.0 error, with and without decay) and differs if the
boundaries are ignored; the swapped kernels' plumbing (fp16 receives float16 q, k, v, beta
and the same g and boundaries; recurrent gets g = 0 for an ungated layer; refusals); the
swap always undone; the exact reference from captured inputs equals Path A's maths via
the sequential reference; flips, position errors, every decision branch on known answers.
11 deliberately broken versions, all caught.

*Cost.* ~30–40 min of GPU time (20 models x 7 paths, two of them float64), GPU idle.

### Fixing the faster path's sequence accuracy — result, 25 October (applied as pre-registered)

`results/ragged_fix.{log,json}`. 20 learned models (seeds 11, 2036, 4044 skipped as
pre-registered); G0, G1, G2 passed on all 20.

**Step 1, our side: CLEAR.**
- Packing: R64 (the real packed path with an exact kernel) agrees with X on **100%** of
  predictions in every model (pooled 1.000000); same sequence accuracy (0.9956).
- Fusion: F agrees with Path A on **100%** of predictions (1.000000).
- Gate flips: **0.0** in every layer, every path, every model.
- So the faster path's sequence loss is not a bug in our packing, fusion or gating: it is
  the bf16 chunk kernel's arithmetic on the packed sequences.

**Where the error enters** (recurrence output against an exact recurrence on the same
inputs, 16 sequences, by quarter of the sequence): R starts with *less* error than Path A
(fewer chunk boundaries) but grows faster, and ends higher in the last quarter in 17 of 20
models (e.g. seed 2038: Path A 3.5e-2 → 9.0e-2, R 2.1e-2 → 1.3e-1) — consistent with the
chunk-density suspect. R16 is about ten times smaller (2–11e-3); **Rrec equals the exact
path** (1.5–1.8e-3, the bf16 output rounding alone).

**Accuracy, pooled over 20 models:**

| | A | F | X | R | R64 | R16 | Rrec |
|---|---|---|---|---|---|---|---|
| sequence | 0.9891 | 0.9891 | 0.9956 | 0.9706 | 0.9956 | 0.9957 | **0.9956** |
| token | 0.99984 | 0.99984 | 0.99993 | 0.99958 | 0.99993 | 0.99993 | **0.99993** |

Step 2: R16 ACCURATE, Rrec ACCURATE (both above Path A).

**Step 3, timing** (seed 12; time / best fixed-4-factor time):

| | 32 x 512 | 32 x 2048 |
|---|---|---|
| R (bf16 chunk, as before) | 0.79 | 0.76 |
| R16 | 0.87 | 0.85 |
| **Rrec** | **0.64** | **0.63** |

**ADOPTED: Rrec** (fla's `fused_recurrent_gated_delta_rule` for the packed recurrence).
Predictions: step 1 clear — held; error growing along the sequence — held in 17/20;
"Rrec accurate but slower" — **wrong: it is the fastest**; R16 in between — accurate, but
the slowest.

**What this changes, narrowly.** With the float32-state kernel, the faster S₅ path keeps
the training path's accuracy — in fact exceeds it (sequence 0.9956 vs 0.9891; token
0.99993 vs 0.99984), because Path A itself loses sequences to bf16 rounding — and runs at
0.63–0.64 of the best fixed-4-factor time. **Not yet checked, before this enters any
document:** (i) the baseline was the chunk kernel; a fixed-4-factor layer given the same
recurrent kernel must be timed too, or the speed-up may partly be the kernel's, not the
gate's; (ii) batch-1 latency and other batch sizes; (iii) timing on more than one model.

### Fair-baseline timing on S5 — pre-registered 25 October, before any run

*Question.* The faster path with the float32-state recurrent kernel ran at 0.63–0.64 of
the best fixed-4-factor time — but that baseline only had the bf16 chunk kernel. How much
of the speed-up is the gate's skipping, and how much the kernel's?

*Setup.* `work/time_s5_kernels.py`. Fixed 4 gets every option the faster path has:
per-factor or fused projections x chunk or recurrent kernel (4 configurations); its
fastest at each size is the baseline. Also timed: fixed 1 (both kernels, the floor), the
gated model with every factor computed (fused Path A, recurrent kernel: the gate's cost
without skipping), and the faster path with each kernel. Sizes batch {1, 8, 32} x length
{512, 2048}, test sequences joined end to end (timing only). Per size: 3 warm-up passes,
then 20 rounds with the 9 configurations in a seeded random order; a pass = 3 forward
calls; median and bootstrap 95% interval of the per-round ratio. Three trained models
(seeds 12, 13, 31). GPU idle (the script refuses otherwise).

*Decision* (per model and size). best4 = the fixed-4 configuration with the lowest
median; ratio = time(faster path, recurrent kernel) / time(best4). **GATE SAVING** if the
whole interval is at or below 0.90 (the original "at least 10% faster" rule); else NO GATE
SAVING. The claim lists the sizes with GATE SAVING on all three models, with their
ratios. Reported either way: the kernel's own effect on fixed 4 (best recurrent / best
chunk), the gate's cost without skipping, the fixed-1 floor, and the 13 October path
(chunk kernel) against the fair baseline.

*Checks.* K1 every configuration labelled "recurrent" called the swapped kernel and
every "chunk" one did not (a call counter) — otherwise a swap that silently failed would
time the chunk kernel under the recurrent label; any K1 failure means no claim from the
run. K2 the gate's mean factors per token on each timed batch (reported).

*Prediction (weakly held).* The recurrent kernel also speeds up fixed 4 at some sizes, so
the fair ratio is above 0.63; at batch 32 it is still GATE SAVING (0.55–0.80), because the
recurrent kernel's work grows with the number of micro-steps (4 per token for fixed 4,
~1.3 for the gate). Batch 1: no prediction (12 heads give the recurrent kernel little
parallel work).

*Self-tests* T1–T3 (CPU): the counted swap counts, reaches the right kernel, leaves the
kernel alone for "chunk" and is always undone; on sleeps of known length, best4 is the
fastest fixed-4 configuration (not a faster non-fixed-4 one), the ratio and its interval,
the kernel effect and the gate's no-skip cost come out right, a 0.95 ratio is NO GATE
SAVING, the 0.90 boundary, the varying and reproducible round order; joined inputs and
their refusals. 10 deliberately broken versions, all caught.

*Cost.* ~10–15 min of GPU time.

### Fair-baseline timing on S5 — result, 25 October (applied as pre-registered)

`results/s5_kernel_timing.{log,json}`. K1 passed at every size on every model (each
"recurrent" configuration called the swapped kernel, each "chunk" one did not). Gate
order on the timed batches 1.25–1.31.

Ratio = faster path with the recurrent kernel / the best fixed-4 configuration (median;
95% interval), seeds 12 / 13 / 31:

| batch x length | seed 12 | seed 13 | seed 31 | best fixed 4 | decision |
|---|---|---|---|---|---|
| 1 x 512 | 1.55 | 1.53 | 1.56 | fused, recurrent | NO GATE SAVING (slower) |
| 8 x 512 | 1.38 | 1.34 | 1.32 | fused, recurrent | NO GATE SAVING (slower) |
| **32 x 512** | **0.74** (0.74–0.75) | **0.77** (0.73–0.81) | **0.76** (0.75–0.81) | fused, recurrent | **GATE SAVING** |
| 1 x 2048 | 1.20 | 1.26 | 1.18 | fused, chunk | NO GATE SAVING (slower) |
| **8 x 2048** | **0.75** (0.75–0.77) | **0.75** (0.74–0.76) | **0.76** (0.75–0.79) | fused, recurrent | **GATE SAVING** |
| **32 x 2048** | **0.71** (0.70–0.72) | **0.72** (0.71–0.73) | **0.70** (0.69–0.71) | fused, recurrent | **GATE SAVING** |

**Claim: GATE SAVING on all three models at 3 of 6 sizes** — the three with at least
16 384 tokens per batch — at **0.70–0.77 of the best fixed-4 time (23–30% faster)**.
- **The kernel's own share.** The recurrent kernel also speeds up fixed 4 (best recurrent
  / best chunk 0.87–0.89 at these sizes), so the earlier 0.63 against the chunk-only
  baseline was partly the kernel's: 0.63 ≈ 0.88 (kernel) × 0.72 (gate).
- **Small batches are slower, by 18–56%.** There, even fixed 1 is about as slow as fixed
  4 (floor 0.73–1.01 of best4): the forward pass is bound by overheads (kernel launches,
  Python), not by the recurrence, so skipping factors cannot pay; and the faster path adds
  its own fixed cost (packing, one host synchronisation, the gate: the gated model with no
  skipping already costs 1.05–1.31x fixed 4). At batch 1 x 2048 the recurrent kernel is
  slow (1.46–1.54x the chunk kernel on fixed 4: 12 heads give it little parallel work).
- The 13 October path (chunk kernel) against the fair baseline: medians 0.86–1.02 at the
  three large sizes (below 0.90 only at 32 x 2048) — the old kernel's speed-up was mostly
  not the gate's against a fair baseline.
- Prediction: fair ratio above 0.63 — held (0.70–0.77); batch 32 GATE SAVING in 0.55–0.80
  — held.

**Accuracy beside it** (results/ragged_fix.json; matched_compute_p0.1.json): the faster
path with this kernel, pooled over 20 gated models, token 0.99993, sequence 0.9956. The
trained fixed-4 model on the same task (one run, seed 666, evaluated with the bf16 chunk
kernel): token 0.99992, sequence 0.9915. Equal at the token level; the sequence-level
comparison is not like for like (fixed 4 was not evaluated with the float32-state kernel,
and is a single run).

**S₅ speed claim, as it may now be stated:** with the float32-state recurrent kernel, the
learned gate's per-token skipping makes inference 23–30% faster than the best fixed-4
layer given the same kernels, at 16k–65k tokens per batch, on three trained models, with
token accuracy equal to fixed 4's; at 2k–4k tokens per batch it is 18–56% slower. RTX
4050 laptop GPU, inference only.

### Revised plan, 8 October (supersedes the October list below)

**Main deliverable:** a DeltaProduct layer whose order is *learned by the halting
gate* and then *built at that size*. Run the working recipe (open start, hard
straight-through gates and the penalty from step 1, first factor always on);
take the largest order the gate gives any token; rebuild the layer with that many
factors and retrain or fine-tune. Factors the gate never uses are removed, so
the saving is real with the existing kernels and can be timed. This is Tier 3 of
Part 7b reached through the Tier-1 gate, and it keeps the gate — the novel
mechanism — at the centre. Per-token savings (Tier 1 proper) remain the
extension and need the ragged kernel (Path B).

1. **Score allocation by the demand model, not transposition length.** Built
   8 October: `exp_gate_train.py --demand model` charges D(g) from the alphabet's
   generated group (S₅ → ℓ; A₅ → 2 for every non-identity element; anything else
   refused). The oracle still uses ℓ. Self-tests T20–T20h.
2. **The A₅ experiment.** The working recipe on the all-5-cycle alphabet (p = 1.0,
   data already generated), scored with `--demand model`; 3-seed probe first,
   20 seeds if it holds. **Prediction:** the gate settles at **2 factors per
   token** — while the same 5-cycles received 4 under the S₅ alphabet. If it does,
   a learned mechanism has independently confirmed the demand model (the
   sharpened Claim 2, item 5b below), and the layer can be rebuilt at 2.
   *Failure condition:* the gate settles at 3 or 4 with accuracy ≥ 0.99 — the
   model is not using the 3-dimensional representation, and the prediction fails
   for learned models. *Known risk:* A₅ at 2 factors escaped the plateau in only
   3 of 10 runs (Part 3c); the gate may never reach 2.
3. **First measured speedup.** Build the layer at the gate's order, train to the
   same accuracy, and time training steps and inference against 4 factors on the
   same GPU. No speedup is claimed before this measurement.
4. **Boxes order sweep** (n_h ∈ {1, 2, 3}), in parallel if Shashwat or the
   institute cluster can take it: does order matter on natural-language state
   tracking at all?
5. **Postponed:** restart-on-failure reliability run; Path B (ragged expansion);
   the open-start / hard-gate attribution of arm B's success.

**Immediate (October) — superseded 8 October, kept as the record**

1. **Find what raised the gate's floor from 1 to 2 factors.** The earliest
   configuration reached the exact 1.2985 on one seed (two runs); nothing since
   has gone below 2.199 (Part 3e). First step: re-run that seed with the
   initialisation it actually had (`--gate_init hf_default`, seeds 1–3,
   γ ∈ {0.03, 0.1}, plus one identical repeat), after a 1-step fingerprint run
   confirms the step-1 losses match `gate_train_p10`. If seed 1 reaches 1.2985
   again, the script edits since are not the cause and the walk continues one
   change at a time (float32 gate, then zero output weight, then bias 4) on the
   same seeds. *Failure condition:* if the fingerprint matches and seed 1 does
   not reach 1.2985, the original was either a draw under bf16 non-determinism
   (the repeat run bounds that) or the result of a script change that was not
   recorded — the script that produced it was not kept.
2. **Measure the gate's escape rate over runs, not seeds**, with ≥20 runs on the
   chosen configuration, reported as k/n with an interval. 3/10 cannot go in a
   document.
   **Stopping rule for the per-token gate (pre-registered 3 October, before the
   hard-gate results).** Usable = closes with the class structure (corr ≥ 0.99)
   *and* hard-gate token accuracy ≥ 0.99. After (i) hard-gate training and
   (ii) penalty from step 1 have each been tried, if the best configuration gives
   fewer than **15 usable runs out of 20 seeds** at p = 0.10, the per-token gate
   goes on hold as a research extension and the main deliverable moves to the
   per-layer learned order (Tier 3, Part 7b). Moving down the ladder must cite
   this rule. Today's figure is 4/20.
2b. **Prototype a per-layer learned order** on the A₅ and S₅ alphabets: does it
   find 2 and 4? It is the shortest path to a *measured* speedup (a layer that
   needs 2 factors is built with 2) and rests directly on the demand model.
3. **Order sweep on Boxes:** DeltaProduct at n_h ∈ {1, 2, 3} (Part 3f). This
   decides whether natural-language entity tracking is a setting where adaptive
   order can help at all.
4. **Re-run the matched-compute sweep over runs** so §5.1's mid-range entries
   mean something, and finish the census on the remaining alphabet cells.
5. **Length extrapolation for `fixed3`** at p=0.1 (§6.1), and the p=0.5 re-run at
   60 000 steps (§6.2). Both need a `--k_test` flag that does not exist yet.

**Next (2–4 weeks)**

5. ~~Wire `budget_loss` into the training loop and train the gate end-to-end.~~
   **Done, and run (Part 3d).** The gate recovered the oracle allocation exactly
   on 1 of 3 seeds and did not close at all on the other 2. The open question is
   no longer *can it* but *why only sometimes*, which is step 5a.

5a. **Make gate closure reproducible.** Suspected cause: the gate starts
   input-independent (output weight initialised to exactly zero) and saturated
   (σ(4.0) = 0.982, derivative 0.018). Two cheap targeted interventions, each a
   flag on `exp_gate_train.py`: `--gate_init_bias 2.0` (derivative 0.105) and
   `--gate_lr` above the model rate. **Stated failure condition:** if closure
   stays at roughly 1 in 3 across ≥5 seeds under both, the initialisation is not
   the cause and the next suspects are the budget schedule (penalty applied
   before the gate is input-dependent) and the straight-through estimator at the
   soft→hard boundary.

5b. **Then the sharpened form of Claim 2:** does the learned gate discover that a
   5-cycle costs 2 under an A₅ alphabet but 4 under an S₅ one? That would be the
   gate inferring a global algebraic property from local statistics — a far
   better interpretability result than "the gate correlates with difficulty".
   Blocked on 5a: a mechanism that fires 1 time in 3 cannot support it.
6. Extend the demand table to S₃, S₄, A₄ empirically; confirm D predicts the
   required `n_h` across groups, not just within S₅.
7. Implement Path B (ragged expansion) and measure real wall-clock.

**Then**

8. Beyond groups: does the demand model say anything about non-group tasks
   (arithmetic, code state)? This is what makes it a paper rather than a note.
9. Write-up. Target venue decided after step 5a.

**Checkpoint discipline.** Each of 1, 2, 5a has a stated failure condition. If
step 1 fails, Claim 2 dies and we fall back to Claim 1 plus the two proved
bounds — which is still a workshop paper. If step 5a fails — closure stays at
roughly 1 in 3 whatever we do to the gate — then what we have is a mechanism
that works sometimes and an oracle that works always, and the honest description
is a theory paper with an oracle, not a mechanism paper. Step 5 itself is no
longer a checkpoint: the gate has been trained end-to-end (Part 3d), and it did
reach the oracle allocation, on one seed in three.

---

## Part 7b — The fallback ladder, pre-registered

**Written 13 September, before the escape census returned.** That timing is the
point: a fallback chosen after seeing the numbers is a rationalisation, and a
fallback chosen before them is a decision. If a later version of this document
moves down this ladder, the trigger below must be the reason given.

### The invariant

The objective is **a linear-RNN layer that spends compute where it is needed
rather than uniformly.** Per-token Householder count is one implementation of
that invariant, not the invariant itself. Implementations that preserve it are
continuations of this project. Anything that abandons it is a different project
and needs saying out loud.

### The tiers

> **8 October:** moved per the stopping rule (Part 3e, "Arm B on 20 seeds"). Tier 1
> is no longer the main deliverable; the main deliverable is Tier 3 with the
> order chosen by the Tier-1 gate (Part 7, revised plan). The trigger was 13/20
> usable against the pre-registered 15/20.

| tier | mechanism | abandon it when |
|---|---|---|
| **1** | Per-token learned gate + ragged expansion (Path B). The current plan. | The base layer still cannot be trained reliably at `K = D` after the β work, curriculum on the demand mixture, and the β=0 warm start; **or** the trained gate cannot approach oracle allocation. **Status 15 Sept:** the second condition is no longer a risk of *possibility* — the gate reached the oracle allocation exactly (Part 3d) — but of *reliability*: it did so on 1 seed in 3. |
| **2** | Per-token gate, scope restricted to the heterogeneous regime `frac(D=max) ≤ 0.25`, stated as a limitation. | The gate cannot be trained at all even where the base layer is reliable. **Status 15 Sept:** this tier's abandon condition has not fired; Part 3d ran inside exactly this regime (p = 0.10) and the gate did train there, intermittently. |
| **3** | Learned order **per layer / per head** instead of per token. | Only if learned order gives no accuracy or cost advantage over the best hand-chosen `n_h`. |
| **4** | The theory plus two tools: a demand predictor that computes the required `n_h` from a task's alphabet before training, and a fix for training at minimal order. No adaptive mechanism. | Never — this is the floor. It is true today. |

Tier 3 deserves more weight than it has had. It still removes the unguessable
knob, which is the *earned* argument of Part 2. It trains far more easily, being
a handful of discrete decisions rather than one per token. And it delivers a real
wall-clock saving **with no ragged kernel at all**, because a layer that learns
it needs two factors is simply built with two — Path B disappears. Less novel
than Tier 1; "works" beats "novel."

### The granularity constraint

The granularity at which demand varies and the granularity at which a GPU
computes efficiently are in tension, and there is nothing usable in between:

- **Per layer** — trivially exploitable, no heterogeneity to exploit.
- **Per token** — all of the heterogeneity, hardest to exploit. Needs ragged
  expansion, and batching dilutes the saving: at batch 1 you get the full
  `E[D]/D_max`, at large batch the cost tends to `max` over the batch. Grouped
  MoE-style kernels could recover it at scale, at MoE-grade engineering cost.
- **Per chunk** — looks like the compromise; the arithmetic kills it. The
  fraction of all-easy chunks is `(1−p)^c`. At p = 0.1 that is 0.43 at c = 8,
  0.19 at c = 16, and **0.001 at c = 64**. This generalises the v1 chunk-routing
  error in Part 9: you would need chunks so small that kernel efficiency dies.

So Tier 1 and Tier 3 are the only viable granularities. Stating this cleanly is
itself a contribution, and it is why there is no Tier 2.5.

### The application objective

**Primary target: chess board-state tracking.** Chosen over the alternatives for
reasons that are checkable rather than plausible:

- Already in this repository (`state_tracking/src/state_tracking/chess/`) and
  benchmarked upstream, so a comparison baseline exists without building one.
- **Not a group task** — which is exactly step 8 above, the thing that makes
  this a paper rather than a note.
- Demand heterogeneity is plausible *and measurable*: most moves change little
  state, captures/castling/promotion change a lot. There is no oracle for a real
  task, so the trained gate is the only instrument that can measure it. This
  makes the gate load-bearing rather than optional.
- Fits in 6 GB.

**Delivery form:** a drop-in `AdaptiveDeltaProduct` layer in
`flash-linear-attention` with a measured benchmark table. A layer someone else
can import is a better answer to "is this real" than a PDF.

**Target statement, at roughly even odds:** *on chess board-state tracking, an
adaptive-order layer matches fixed-`n_h` accuracy at lower measured per-token
cost, and delivers measured wall-clock savings at batch 1.* The batch-1
qualifier is load-bearing, not a hedge — see the granularity constraint.

**Ruled out now, to stop them being reached for later:** financial time series
and general language modelling. Our own result says demand comes from alphabet
closure, and neither has an analogue, so citing the demand model in their support
would borrow credibility we have not earned. Robotics only for genuinely
discrete state.

---

## Part 8 — Repository map

```
work/exp_beta_reachability.py   rank bound, parity wall, representation width,
                                approximation floor.  float64 CPU, ~30s
work/exp_minimal_order.py       the demand model D(g); builds A5's icosahedral
                                rep and verifies it.  float64 CPU, ~10s
work/exp_group_closure.py       THE decisive test of Claim 2.  GPU, ~2.5h
work/exp_matched_compute.py     Checkpoint A, five arms.  GPU
work/exp_error_forensics.py     F1–F5 failure-mode forensics.  GPU
work/gen_heterogeneous.py       heterogeneous-demand data generator
work/reference_deltaproduct.py  float64 oracle for the algebra
src/gating.py                   the halting gate (7 tests)
src/oracle_order.py             ground-truth order injection
flash-linear-attention/…        patched upstream (adaptive_order, external_gates,
                                config forwarding); .orig backups kept
results/                        every log and JSON referenced above
results/escape_census.json      33 runs of the stock architecture; the file that
                                retired the training wall.  Part 3c
AUDIT.md                        methodology audit, 13 Sept: which claims survive,
                                which errors caused which retraction, evidence rules
work/make_figures.py            regenerates every report figure from results/*.json
                                and nothing else; --selftest = 10 known-answer
                                checks, --report writes caption-free versions
work/check_gate_precision.py    bf16 vs fp32 saturation of the halting sigmoid
results/gate_gamma_sweep.json   the penalty-weight sweep, 10 runs.  Part 3e
results/boxes_hybrid.json       Boxes entity-tracking results.  Part 3f
report_sept2026/                mid-semester report, LaTeX sources + PDF (54 pp)
presentation/                   mid-semester presentation (.pptx, 22 slides)
figures/, figures/report/,      figure outputs: screen, report (no titles),
figures/slides/                 and slide-sized (larger fonts)
```

Every experiment file carries a `--selftest` or self-test block that runs
against data whose answer is known in advance. This is not ceremony — §9 lists
three occasions where the self-test caught a wrong result before it reached this
document.

---

## Part 9 — Errors caught, and how

Kept because the methodology is part of the contribution, and because two of
these were mine and stated as findings before being caught.

| what | how it was caught |
|---|---|
| v1 chunk-routing arithmetic: 96% of chunks hard ⟹ zero savings | worked the arithmetic |
| 246 MB of CSVs nearly committed (inline comment in `.gitignore`) | dry run |
| Straight-through estimator grouping bug rounded off {0,1} | unit test |
| `--k_test=512` measured extrapolation, never training-length accuracy | my error; caught on review |
| Per-batch metrics dominated training: 22m34s → 4m47s | profiling, after 3 wrong hypotheses |
| **Overclaimed "trained end-to-end" on the departmental form** | **user caught it** |
| Missed DeltaProduct's own future-work paragraph | user prompted a re-read |
| Vocabulary trap: 34 inputs, 120 targets, 86 would become UNK | caught before running |
| `β = 2·sigmoid(raw)` cannot attain β=2 — exact solutions floored at 1e-5 | disagreed with theory |
| Unconstrained β optimizer collapsed to A=I and reported it as "unreachable" | residual was exactly ‖P−I‖ |
| Persistence statistic scored a purely absorbing world at 3.7× | synthetic self-test |
| Heterogeneity measure counted the identity, hiding the A₅ case | reviewing the output |
| **Parity split divided by an empty set**, printing a "parity wall" verdict on every A₅ cell — where A₅ *is* the even half and there are no odd targets to fail | reading the output; all three arms printed it, including one scoring 0.9997 |
| **Evaluation truncated at 384 sequences** by a stray `break`, while every other script reports over 2000 — headline numbers not comparable | reading the eval loop |
| **Verdict compared against a hardcoded number from another script**, printing FIXED 0.1259→0.9997 when the in-run control had scored 0.9598 | in-run control disagreed with history |
| **Verdict compared medians of a bimodal distribution** — 2/4 vs 1/4 escapes gave medians of 0.98 vs 0.12, an apparent 8× effect at Fisher p = 1.000 | replaying real numbers through the new code |
| **Single-seed reporting on bistable cells**, plus a noise floor measured on a stable cell and applied globally. Cost: the Part 3b training wall, now retracted | the escape census, 33 runs |
| **`par[pred]` indexed a 120-entry table with a 127-token argmax** — device-side assert killed the census at run 34 of 40. Latent in the original; exposed ~5× by fixing the truncated eval above | the crash; now self-test T5b |

The last four are the ones that matter: **two would have put a false result in
this document**, and both were caught by a test written to fail rather than to
pass. The reachability question was consequently re-answered by explicit
construction instead of by optimizer, so no claim in Part 2 depends on an
optimizer converging.

---

## Part 10 — Position in the literature

- **DeltaProduct** (NeurIPS 2025) introduces the `n_h`-Householder product and
  fixes `n_h` globally. Its own future-work section names per-token adaptive
  order and cites Graves' ACT. We are downstream of that sentence, and doing it
  carefully is a workshop paper, not a NeurIPS one. **The demand model is not
  in that paper and contradicts its implicit cost accounting.** That is the part
  that is ours.
- **Shakerinava, Khavari, Ravanbakhsh & Chandar** (ICLR 2026), *The Expressive
  Limits of Diagonal SSMs for State-Tracking* (arXiv:2603.01959), study
  input-dependent complex-valued diagonal SSMs and report that multi-layer
  models **often fail to learn state tracking for non-Abelian groups**. Our
  `fixed2` at p=1.0 is that phenomenon for Householder products.
- **PonderNet / ACT** (Banino 2021; Graves 2016) supply the monotone halting
  mechanism. Off the shelf, and we say so.
- **Cartan–Dieudonné** gives the classical reflection count; the representation-
  dependence of the *minimum* over faithful representations is the step nobody
  in this line has taken.

**On the recombination question.** The mechanism — DeltaProduct's recurrence
plus ACT-style halting — is a recombination, and every part is off the shelf.
What makes it research is that building it forced a problem nobody had: the
parity wall, and then the demand model. Part 3 was not available by reading; it
came out of an experiment that contradicted a theorem we had just proved. That
is the part to write the paper around.

---

*Next action, as of 13 September evening.* The β-parametrisation question is
**suspended, not answered.** It was built on the seven-failure list, and the
escape census dissolved that list — every cell escapes some of the time, so there
may be nothing about β to fix. Comparing β modes properly now needs ~20 runs per
arm on a cell we have first shown to be reliably hard, and we do not have one.

What actually comes next is the confound in our longest run (Part 3c):

```
# Does S5_c5_t1:4 escape at CONSTANT learning rate?
# The 250k run that "ruled out under-training" used cosine decay to zero,
# which makes late escape impossible. ~2.1 h.
nohup python -u work/exp_beta_param.py --cells S5_c5_t1:4 --modes sigmoid:0 \
  --seeds 1,2,3,4,5,6,7,8,9,10 \
  --out results/escape_s5t1.json > results/escape_s5t1.log 2>&1 &

# Then finish the interrupted census cell. ~1.5 h.
nohup python -u work/exp_beta_param.py --cells S5_c5_t:4 --modes sigmoid:0 \
  --seeds 4,5,6,7,8,9,10 \
  --out results/escape_census2.json > results/escape_census2.log 2>&1 &
```

If `S5_c5_t1:4` escapes at constant LR, the longest run in this project was
measuring its own learning-rate schedule, and the open problem for the whole
project becomes **"why is the escape rare, and can it be made reliable"** — an
optimisation question, which is more tractable than the representational one we
thought we had.
