# Adaptive-Order State Tracking — project brief and working agreement

Paste this whole file as the **Project instructions** for a new Claude Project,
or as the first message of a new chat. Everything after it can be normal
conversation.

---

## 0. How to work with me

These are not generic preferences. Each one was learned from a specific mistake
in the previous sessions, and repeating any of them costs real GPU hours.

**Verify before claiming.** Never describe intended work as completed. I once
wrote "trained end-to-end with a compute-budget penalty" on a departmental form
when the gate had never been trained. If something is planned, say planned. If a
number came from an oracle rather than a learned model, say so every time.

**When something fails, try to fix it — do not convert it into a finding.** My
strongest instinct is to write up a failure as "an interesting negative result."
That is paper posture, not building posture. Seven training failures got filed as
seven pieces of evidence when they were one bug worth attacking. Push back on me
when I do this.

To be precise, because the rule is easy to over-apply: **record and analyse every
negative result — that part is good.** What is forbidden is the next step, "this
has not been reported before, therefore it is a contribution." The novelty claim
is what smuggles a bug into a paper. **Novelty in a negative result is weak
evidence of a contribution and moderate evidence of a setup problem:** if a
well-resourced field has not reported it, the likeliest explanations in order are
(1) our setup, (2) nobody tried, (3) a real finding. On 13 September a sentence in
`PLAN.md` claiming a training wall was "resource-independent, reproducible, and
not reported anywhere in the literature" turned out to be (1) — every cell was a
single draw from a bistable outcome. See `AUDIT.md`.

**Attribution before mechanism.** Before blaming a published architecture, rule
out my own experimental setup. I built a theory about a flaw in DeltaProduct's
β parametrisation while running at 1/10th the training steps of the only
configuration confirmed to work. Ask "is this our fault?" early and take the
answer seriously.

**Do not read trends from a handful of noisy points.** I called a loss curve
"still descending" from four logged values; 60 000 further steps showed it was
flat oscillation. If a claim rests on a trend, plot or measure it.

**Separate notional from measured.** Our compute savings are *required work per
token*, not wall-clock. The adaptive arm took 756s against 761s for the
full-cost arm — essentially identical, because unused factors are still computed
and multiplied by zero. Never state a speedup we have not measured.

**Write tests designed to fail.** Every experiment file should have a self-test
that runs against synthetic data whose answer is known in advance. Two wrong
results were caught this way before they reached a document; two others were not
and had to be retracted.

**Tell me what files you are changing, before or as you change them.** Do not
edit silently.

**Own errors plainly.** Acknowledge, correct, move on. No spiralling apology, no
defending a position that the data has killed.

**Git:** branches are wanted (a teammate will join). Commit messages must be
understandable to someone outside this project — **no internal phase numbering
or codenames.** GitHub account: `Shukla070`.

**Long runs are fine.** 10–12 hour GPU runs overnight are acceptable and
expected. Do not design experiments so lean that the baseline itself fails —
that was a real error and it wasted days.

---

## 1. The project

**Title:** Adaptive-Order State Transitions in Linear RNNs for Efficient State
Tracking
**Who:** Utkarsh Shukla, 4th-year B.Tech AI, NITK Surathkal. Guide: Anand Kumar M.
Year-long major project (IT448), Sept 2026 – Apr 2027.

**The base method.** DeltaProduct (Siems et al., NeurIPS 2025, arXiv 2502.10297)
is a linear RNN whose state-transition matrix at each step is a product of `n_h`
generalized Householder factors. More factors = more expressive = more compute.
`n_h` is a **single global hyperparameter fixed by hand** and applied identically
to every token.

**The mechanism we are building.** Make that count **per-token and learned**: a
small monotone halting gate decides how many factors each token actually needs,
and a budget penalty pushes it to be frugal.

**The real goal.** A new mechanism that is genuinely *used* for something. A
publication would be evidence the work is real, not the objective. Do not
optimise for publishability at the expense of building something that works.

**Final objective.** A DeltaProduct-family layer that allocates computation per
token, trains reliably, and delivers measured wall-clock savings at equal
accuracy — usable as a drop-in in a real sequence model.

---

## 2. Why anyone would want this

The honest, earned version — not the aspirational one.

**The strongest argument, and it follows from something we proved:** `n_h` is a
hyperparameter **nobody can choose correctly**. We proved the required value
depends on global structure of the input distribution, not on anything you can
read off individual tokens. On a real task you cannot compute it — so you
over-provision and waste compute, or under-provision and fail silently. A layer
that learns its own order removes an unguessable knob. This is true today and
needs no speedup to be true.

**The efficiency argument, once Path B exists:** cheaper inference for any
deployed sequence model using these layers — LLM serving, edge/mobile, real-time
streaming. Currently *unearned*: we have no measured speedup.

**Plausible but untested:** time-series regime detection, robotics/control,
translation. Do not claim these. Our own finding says per-token demand comes
from closure properties of the whole symbol alphabet, and language has no
obvious analogue.

---

## 3. Where the work actually stands

### Proved (machine precision, reproducible)

- **`K ≥ ℓ` is forced by rank.** A product of K rank-1 updates satisfies
  `rank(A − I) ≤ K`, and `rank(P − I) = ℓ(P)`. True for every β.
- **The parity wall.** With β pinned at 2, `det(A) = (−1)^K`, so only
  `K ≡ ℓ (mod 2)` is reachable. Therefore **the gate must scale a learnable β,
  not mask a fixed one.** Verified by construction on all 120 elements of S₅,
  worst error 1.17e-15.
- **The demand is not the transposition length.** `D(g) = min over faithful
  representations ρ of ⟨alphabet⟩ of rank(ρ(g) − I)`. A 5-cycle costs **4 inside
  S₅** and **2 inside A₅**, because A₅ has a 3-dimensional faithful
  representation (the icosahedral rotation group) and S₅ has none. Verified by
  building that group from geometry: order 60, correct element-order profile,
  explicit isomorphism, every non-identity element rank 2, two-reflection
  factorisation to 1.31e-15.
- **Consequence:** the cost of a token is a property of the **whole alphabet**,
  not the token. Adding one transposition — the cheapest symbol that exists —
  to an alphabet of 5-cycles doubles the cost of every 5-cycle.

### Measured

- **DeltaProduct reproduced.** S₃ matches the published pattern. S₅ at `n_h=4`:
  token accuracy **0.864 at length 512** after training at 128.
- **Matched-compute sweep** (single seed, under-resourced — see caveat): the
  oracle solves at cost 1.15 / 1.30 / 1.75 per token where the cheapest solving
  fixed order costs 4, at p = 0.05 / 0.10 / 0.25.
- **Group-closure grid**, five alphabets. At n_h=3: A₅ alphabets give
  0.9991, 0.9994, 0.9952; S₅ alphabets give 0.1094, 0.0307. Perfect separation
  by the predicted variable. Alphabet **size** ruled out — the 44-token A₅
  alphabet is the easiest arm, the 25-token S₅ alphabet fails.
- **D = 2 confirmed empirically:** `A5_c5_3c` trained at **n_h = 2** (0.9510).
  Impossible if ℓ were the demand.

### The open problem — restated 13 September

**Not a wall. A low-probability escape from a loss plateau.**

The earlier version of this section said the architecture *cannot* train when
most tokens require the maximum order, citing "0 of 4" cells. That was wrong, and
the way it was wrong is the most useful thing in this document.

Every cell we had called a training failure is **bistable**: it escapes the
plateau some fraction of the time and sits at chance the rest. 33 runs of the
unmodified architecture, 20 000 steps, constant LR, data fixed:

| cell | historical (n=1) | escapes / runs | rate |
|---|---|---|---|
| `A5_c5:2` | 0.1259 | 3 / 10 | 0.30 |
| `A5_c5_dt:2` | 0.0443 | 2 / 10 | 0.20 |
| `A5_c5_dt:4` | 0.0429 | 1 / 10 | 0.10 |
| `S5_c5_t:4` | 0.0706 | 1 / 3 | 0.33 |

**`S5_c5_t:4` seed 3 reached token 0.9994, sequence 0.9415, with parity 0.9995
even / 0.9994 odd.** That is a high-demand S₅ cell (frac(D=max) = 0.71) solving
the task and resolving the parity bit completely. The wall had a counterexample
inside it.

Every one of the historical numbers was a single unlucky draw. The "seven
failures" premise — which the whole β-parametrisation theory was built on —
does not survive.

**The confound this exposes.** The 250 000-step run that "ruled out
under-training" used warmup + **cosine decay to zero**, which makes a late escape
impossible. The census runs held LR constant and could escape at any step; some
escaped as late as 15–20k. So that run may have been decayed into the plateau
rather than trained out of it, and **"under-training is ruled out" is withdrawn**
pending a constant-LR rerun.

**Why this is better news than the wall was.** An optimisation-basin problem is
more tractable than a representational limit, and it turns the open question into
one worth a project: *why is the escape rare, and can it be made reliable?*
Anyone using DeltaProduct at minimal order has this problem.

**Still true:** adaptivity only helps when demand is heterogeneous, and that
regime trains readily.

### Not done

- Gate **never trained end-to-end**. Every adaptive number uses an oracle that
  reads ground truth. `budget_loss` exists, tested, not wired in.
- **Path B** (ragged expansion) not built. No real speedup exists.
- Most tables are **single-seed**, and single seeds are now known to be
  meaningless on bistable cells. They need re-running as escape fractions before
  any of it goes in a document.
- The **0.545 A₅ ceiling** and `effective_cost` / `predicted_demand` are derived
  numbers with no self-test. Both are load-bearing. See `AUDIT.md` §4.

---

## 4. The immediate next step

The β-parametrisation question is **suspended, not answered.** It rested on the
seven-failure list, and the census dissolved that list — there may be nothing
about β to fix. Comparing β modes properly needs ~20 runs per arm on a cell first
shown to be reliably hard, and no such cell is currently known.

What comes next is the learning-rate confound above:

```bash
# Does S5_c5_t1:4 escape at CONSTANT LR? ~2.1 h
nohup python -u work/exp_beta_param.py --cells S5_c5_t1:4 --modes sigmoid:0 \
  --seeds 1,2,3,4,5,6,7,8,9,10 \
  --out results/escape_s5t1.json > results/escape_s5t1.log 2>&1 &

# Then finish the census cell interrupted by the crash. ~1.5 h
nohup python -u work/exp_beta_param.py --cells S5_c5_t:4 --modes sigmoid:0 \
  --seeds 4,5,6,7,8,9,10 \
  --out results/escape_census2.json > results/escape_census2.log 2>&1 &
```

`python work/exp_beta_param.py --selftest` first — it is instant and covers the
diagnostics that have produced three retracted results.

If `S5_c5_t1:4` escapes at constant LR, the longest run in this project was
measuring its own optimiser schedule.

---

## 5. Repository

`C:\D\B.Tech\Major Project\DeltaProduct` on the user's machine (WSL path
`/mnt/c/D/B.Tech/Major Project/DeltaProduct`). Activate with `mp` alias, venv at
`.venv`.

| file | what | GPU |
|---|---|---|
| `PLAN.md` | master record: claims, evidence, plan, errors caught | — |
| `AUDIT.md` | **read this second** — which claims survive the 13 Sept methodology audit, and the evidence rules that came out of it | — |
| `GLOSSARY.md` | every term and symbol, written for a newcomer | — |
| `work/exp_minimal_order.py` | verifies the demand model D; builds A₅'s 3-D rep | no |
| `work/exp_beta_reachability.py` | the rank bound and parity wall, by construction | no |
| `work/exp_group_closure.py` | alphabet-closure grid; `--warmup --cosine --eval_every` | yes |
| `work/exp_beta_param.py` | the current fix attempt + β/det/parity diagnostics | yes |
| `work/exp_matched_compute.py` | fixed orders vs oracle adaptive | yes |
| `work/exp_error_forensics.py` | failure-mode forensics F1–F5 | yes |
| `work/patch_beta_mode.py` | patches the layer for swappable β; `--revert` | — |
| `src/gating.py` | the monotone halting gate (7 tests) | no |
| `src/oracle_order.py` | ground-truth order injection | no |
| `results/` | every log and JSON referenced above | — |
| `flash-linear-attention/` | patched upstream; `.orig` backups kept | — |

Most experiment files support `--selftest`.

---

## 6. Environment traps

Hardware: RTX 4050, **6 GB VRAM**, WSL2 on Windows, repo on `/mnt/c`.

- **`tail -f` does not work on `/mnt/c`** — no inotify on DrvFs. It prints
  `tail: No data available` and then silently shows nothing while the job runs
  fine. Use `watch -n 120 "tail -n 8 <logfile>"`.
- **Launch long jobs with `nohup python -u ... > log 2>&1 &`.** The `-u` matters
  or the log stays empty for an hour and looks like a hang. **Run the block
  once** — a job has been double-launched before. Verify with `nvidia-smi`.
- **Do not put `tmux new` in the same paste as the command** — tmux takes the
  terminal and the rest of the paste is lost. Two separate steps, or use nohup.
- **6 GB VRAM:** batch 256 is the ceiling at 12 heads × 32 with `n_h=4`. WSL
  does **not** OOM when you exceed VRAM — it silently spills to system RAM over
  PCIe and runs ~8× slower. A mysteriously slow run means check memory.
- **bf16 kernels are non-deterministic, and there are TWO noise floors.** On a
  cell that reliably solves, identical config and seed gave 0.9696 and 0.9740 —
  floor ≈0.005 token / 0.05 sequence. **On a bistable cell that figure is
  meaningless:** `A5_c5:2` at the *same seed* gave 0.1177 and 0.9598, a spread of
  0.84. Applying the stable-cell floor to bistable cells is what licensed every
  single-seed claim we have had to retract.
- **Outcomes are bimodal**, not Gaussian — a run lands near 0.999 or near 0.04,
  rarely between. Report the **fraction of runs that escape the plateau**, not
  a mean. And **seed is not the unit of replication** — hold the data seed fixed,
  vary the run, and report over runs. Telling a 25% escape rate from a 75% one
  needs ~20 runs per arm.
- **Escape is abrupt and late-arriving.** Loss sits at chance for thousands of
  steps and then drops two orders of magnitude inside one logging interval —
  observed anywhere from <5k to 15–20k steps. A 20 000-step budget therefore
  *undercounts* escapes, and **a decaying LR schedule can prevent a late escape
  entirely.** Use constant LR when measuring escape rates.
- `chunk_gated_delta_rule` asserts non-fp32 input — bf16 only.
- `generate_data.py` silently no-ops if the file exists; pass `--overwrite`.
- `make_data` in `exp_group_closure.py` is a Python loop: ~15 min of silence for
  300k sequences before training starts. Not a hang.

---

## 7. What to do first in the new chat

Read `PLAN.md` and `GLOSSARY.md` from the repo if the session can reach the
user's computer. Then confirm your understanding of Section 3 (what is proved
versus measured versus open) before running anything — particularly that the
gate has never been trained and that no measured speedup exists.

Then run the β experiment in Section 4.
