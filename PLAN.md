# Adaptive-Order DeltaProduct — Master Plan and Record

**Version 6** · 11 September 2026 · supersedes v1–v5 entirely
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
| Real wall-clock saving (ragged expansion) | **not done** | all costs are notional |

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

**Immediate (this week)**

1. **Run `exp_group_closure.py`.** 5 alphabets × 3 orders, ~2.5h. This is the
   decisive test of Claim 2. Predictions are printed before the results, and
   the script names its own failure condition.
2. **Length extrapolation for `fixed3`** at p=0.1 (§6.1). Cheap, decisive.
3. **Re-run p=0.5** at 60 000 steps to close §6.2.
4. **Three seeds** on every headline cell (§5.3).

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
