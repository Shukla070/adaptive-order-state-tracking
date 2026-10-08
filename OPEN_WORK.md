# Open work — adaptive-order state tracking

Standing queue. Updated 1 October 2026, after the mid-semester report and
presentation were finished.

---

## Where the project stands

**Proved, and not contingent on anything.** The rank bound `K ≥ ℓ`, the parity
bound under pinned β, and the order requirement
`D(g) = min over faithful ρ of rank(ρ(g) − I)` — all by explicit construction in
float64, worst error ≈1.3e-15. A 5-cycle costs 4 factors under an S₅ alphabet
and 2 under an A₅ one, so a token's cost is a property of the whole alphabet.

**Measured with a reference allocation (the oracle, which reads a lookup table
derived from that analysis).** At p = 0.05 / 0.10 / 0.25 it matches the accuracy
of fixed `n_h = 4` at 1.150 / 1.300 / 1.752 factors per token. Single runs; the
mid-range entries of that sweep are not capability measurements.

**Measured with the learned gate.** 7 of 49 runs recover the demand structure,
from 4 distinct seeds; 3 of 10 in the penalty-weight sweep (2 of 5 seeds), at
2.199 factors per token and token accuracy 0.9998. Under-allocation is rare:
one seed (16, 2 October), in runs where the task itself was not learned.
Accuracy is ≥0.998 in every run of the penalty-weight sweep, but not across all
49: five runs fall below, the lowest at 0.7944. The exact allocation (1.2985)
was reached only in the earliest configuration, on one seed, from a gate that
started almost shut (the pre-repair initialisation) — see PLAN Part 3e,
corrections of 1 October.

**Measured outside groups.** On Boxes, fixed `n_h = 2` DeltaProduct reaches
0.968 exact match after one operation against 0.312 for equal-size attention,
and extrapolates to 0.818 at five or more operations. No gate, no order sweep.

**Not established.** Any wall-clock saving (all costs are required work per
token; the oracle ran 756 s against fixed4's 761 s); reliable gate training;
anything about adaptive order in natural language.

---

## Evidence rules that still bind

- **Escape fraction, never a mean,** on bistable cells. Report `k/n` with an
  interval and the sorted list.
- **Seed is not the unit of replication.** bf16 kernels are non-deterministic;
  identical config and seed gave 0.9598 and 0.1177. Hold the data fixed, vary
  the run.
- **A derived statistic gets a known-answer test before it enters a document**,
  and the test must be shown to fail against the wrong implementation.
- **Notional vs measured:** our savings are required factors per token. Never
  state a speedup that has not been timed.
- `work/make_figures.py --selftest` (10 checks) must pass before any figure is
  used; it refuses to draw an escape fraction on a cell that is not bimodal.

---

## Next, in order

**Decided 8 Oct — stopping rule applied, not amended** (PLAN Part 3e, Part 7).
13/20 exact stands as the per-token result. Main deliverable is now a layer whose
order is learned by the gate and then built at that size, so the saving can be
timed with existing kernels; per-token savings become the extension.

**A. Demand-model scoring** — *built 8 Oct*: `--demand model` (S₅ alphabet → ℓ;
A₅ alphabet → 2 for every element; other groups refused). Self-tests T20–T20h;
numbering verified against both data files.
**B. A₅ experiment** — working recipe at p = 1.0 with `--demand model`; 3-seed
probe, then 20. Predicted: 2 factors per token. Fails if it settles at 3 or 4.
**C. Rebuild at the gate's order and time it** — the first measured speedup.
**D. Boxes order sweep** — parallel, Shashwat or cluster.
**Postponed:** restart-recipe reliability run; Path B; attribution of arm B.

*Everything below is the record of how we got here.*

**1. Find what raised the gate's floor from 1 to 2 factors.** The earliest
configuration (`gate_train_p10.json`) reached the exact 1.2985 on seed 1 (two
runs). Nothing since has gone below 2.199, across four later configurations and
37 further runs — but no later run used seed 1 with the initialisation it had.

  a. **Done 1 October — 20/20 MATCH.** Fingerprint (1 step, ~1 min): `--gate_init hf_default --steps 1
     --fingerprint results/gate_train_p10.json`, seeds 1–3, and the same against
     `gate_bias40_more.json`, seeds 4–10. Every line must read MATCH before
     anything long is run.
  b. Reconstruction (7 runs, ~2 h): `hf_default`, seeds 1–3, γ ∈ {0.03, 0.1},
     plus seed 1 at γ=0.03 a second time, dense warmup logging.
  c. If seed 1 recovers 1.2985: walk forward one change at a time on the same
     seeds — float32 gate, then zero output weight, then bias 4.

*Failure condition:* fingerprint MATCH and seed 1 does not recover 1.2985 — the
original was a draw or an unrecorded script change, and there is no regression
to find.

**Result (2 October) — closed, no regression.** `gate_hfinit_p10.json`: the
identical earliest configuration gives 1.2985 once and 2.199 once on seed 1 at
γ = 0.1, and 2.199 / 1.4778 at γ = 0.03. Seed 1 closes 4/4, seeds 2–3 0/8. Which
seeds close is decided in the first ~1 000 warmup steps by whether the gate
saturates fully open (PLAN Part 3e, "Rebuilding the earliest configuration").

**2. Establish the gate's success rate over runs.** ≥20 runs on the chosen
configuration, reported as `k/n` with an interval. 3/10 cannot go in a document.
Since the seed decides closure, replicate over **distinct seeds**, not repeats.
First lever tested (2 October): a logit wall. It removed saturation (0/20) but
did not raise closure (8/20 vs 10/20 without) and slowed learning, so it is
dropped. Two findings from that run now lead (PLAN Part 3e, "The logit wall"):
**(a)** most closed runs lose accuracy only at hard-gate evaluation — fix first:
evaluate soft and hard, and train with hard (straight-through) gates once the
penalty has ramped; **(b)** the warmup locks factors open by more than
saturation — next lever: penalty from step 1.
*Run 3 Oct (6 seeds, `gate_hard_6seeds.json`):* hard gates from step 12 000 did
not raise usable runs (1/6 vs 2/6) and shocked training at the switch (seed 13:
loss 0.004 → 0.937, ended 0.983); only 4/12 runs closed by step 12 000, so the mismatch test
had no collapsed runs to explain. Seeds do not reliably repeat closure. PLAN Part
3e, "Hard gates late in training".
Candidate levers once the baseline is known: when the penalty switches on and
how it ramps, the gate's initial bias, PonderNet-style halting (every step
receives gradient), and a curriculum on the demand mixture.

**Stopping rule (pre-registered 3 Oct, PLAN Part 7):** after hard-gate training
and penalty-from-step-1 have each been tried, fewer than 15/20 usable runs
(corr ≥ 0.99 and hard token ≥ 0.99) at p = 0.10 puts the per-token gate on hold
and makes the per-layer learned order the main deliverable. Today: 4/20.
**Penalty-from-step-1 probe (6 Oct, 3 seeds):** arm A 0/3 usable — stopped by the
pre-registered rule; arm B (open start, hard gates and penalty from step 1, first
factor always on) **3/3 at the exact 1.2985** with accuracy ≥ 0.9992 — earns the
full 20 seeds. A same-seed smoke of B had not escaped by step 2 500, so this is
still a race. PLAN Part 3e, "Penalty from step 1".
**Arm B on 20 seeds (7 Oct, `gate_early_B20.json`): 13/20 usable, all 13 exact
(1.2985, token ≥ 0.9986).** Every failure under-allocates 5-cycles and never
learns the task; the outcome is visible at step 500. **Stopping rule triggered**
(13 < 15). Decision pending: apply it (per-layer becomes the main track) or
record an amendment with its reason before any further run. PLAN Part 3e,
"Arm B on 20 seeds".
*Built 6 Oct, not yet run:* `work/run_gate_early_penalty.sh` — 20 seeds, two arms:
(A) pre-repair start, soft gates, penalty from step 1; (B) open start, hard
straight-through gates and penalty from step 1. ~11 h. Both arms keep the first
factor always on (`--min_order 1`, new 6 Oct): without it, a 500-step smoke of
arm A closed every factor by step 250 (order 0.07, loss at chance) — penalty
from step 1 drives the gate shut before the task is learned, and a layer at
order 0 writes nothing.

**2b. Per-layer learned order prototype** on A₅ / S₅ alphabets — should find 2
and 4. Shortest path to a measured speedup.

**3. Order sweep on Boxes:** (run in parallel if Shashwat or the institute
cluster can take it) DeltaProduct at `n_h ∈ {1, 2, 3}`, three seeds
each. This decides whether natural-language entity tracking is a setting where
adaptive order can help at all. If `n_h = 1` already matches `n_h = 2`, it is
not. Then train the gate there and check whether operation tokens receive more
factors than description tokens.

**4. Ragged expansion, and a measured wall-clock number.** There is now a real
learned allocation to exploit (2.199 against a ceiling of 4). A per-level
capacity router in the spirit of Mixture-of-Depths is the natural design,
because static tensor shapes are the whole reason it can pay.

**5. Re-run the matched-compute sweep over runs**, so §5.1's mid-range entries
mean something, and finish the census on the remaining alphabet cells.

**6. Known-answer tests for `effective_cost` and the A₅-ceiling calculation.**
Both are derived statistics that entered documents without one, and two
statistics of that class have already turned out wrong.

**7. Re-run S₃** so the reproduction has a log behind it (the console log was
not kept; only a checkpoint directory survives). Five minutes.

**8. Length extrapolation for `fixed3` at p = 0.1**, and the p = 0.5 re-run at
60 000 steps. Both need a `--k_test` flag that does not exist yet, plus a
512-length heterogeneous dataset.

---

## Broader applications to try (discussion, 3 Oct)

Exploring is not claiming: try these, claim only what is measured.

- **bAbI-style / Boxes-style entity tracking QA** — closest to what works; cheap,
  trainable from scratch on 6 GB.
- **SQuAD** — mainstream extractive QA. Mostly recall, little state change, so
  expect little from order; still a useful check of the layer on ordinary text,
  and the gate's allocation shows which tokens it treats as expensive.
- **Situation tracking over update streams (e.g. a disaster-response RAG
  system)** — reports arrive as updates (road closed, shelter full, resource
  moved); a query needs the *current* state, which plain retrieval gets wrong
  when facts go stale. Most tokens are filler and a few are updates:
  heterogeneous demand, the regime where per-token order should pay. Start with a
  synthetic Boxes-like generator in that domain.
- **Building occupancy from WiseNET** (Marroquin et al., Data in Brief 2019;
  6 indoor cameras, 77 annotated people tracks, building model with rooms and
  doors). Not as a vision task: derive the event stream (person enters/leaves a
  space, groups merge/split) from the ground-truth annotations and ask "who is
  in room X now". Real movement data for Boxes-style tracking; links to the
  disaster idea (who is still inside during an evacuation). Too small to train
  on: use as a real test set, train on a simulator of the same building. First
  check: do the annotations give identities across cameras?
- **Chess board-state tracking** — already the Part 7b application target.

## Deferred deliberately

**The additive input pathway** (arXiv:2609.18966). Worth testing, but it targets
the high-demand regime, which the census showed is not the blocker it was
believed to be and which adaptivity does not target anyway.

**SCONE, ProPara, NPN-Cooking.** The right second and third domains after Boxes,
but only once the gate is reliable and the Boxes order sweep says the order
matters there.

---

## Deliverables that exist

- `report_sept2026/` — mid-semester report, LaTeX sources and PDF, 54 pages.
  Chapters 1–5 total 30 pages. Front matter carries both authors.
- `presentation/Major_Project_Mid_Presentation.pptx` — 22 slides with speaker
  notes, matching the report's numbers.
- `work/make_figures.py` — every figure, regenerated from `results/*.json`.
  `--report` for the report, and a slide variant with larger fonts in
  `figures/slides/`.
- `results/boxes_hybrid.json` — the Boxes numbers, transcribed from the notebook
  on the `nlp_experiment` branch. **Ask Shashwat to commit the run's own
  `hybrid_results.json`**, which also carries the training curves.

### Still to do on the documents

- The report and the deck both state the results above correctly, including the
  caveats. If any number changes, regenerate figures first, then edit the
  chapter, then rebuild.
- Report placeholders: the guide's designation on the certificate, and the
  declaration date.
- `PROGRESS_REPORT.md` predates the γ sweep and Boxes; `PLAN.md` v7 is the
  current record.
