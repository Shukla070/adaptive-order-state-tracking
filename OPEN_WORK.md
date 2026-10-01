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

**Measured with the learned gate.** 7 of 49 runs recover the demand structure;
3 of 10 in the penalty-weight sweep, at 2.199 factors per token and token
accuracy 0.9998. No run ever under-allocates, and accuracy is ≥0.998 in every
run including the failures. The exact allocation (1.2985) was reached only in
the earliest configuration.

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

**1. Find what raised the gate's floor from 1 to 2 factors.** The earliest
configuration (`gate_train_p10.json`) reached the exact 1.2985 twice in 12 runs.
Nothing since has gone below 2.199, across four later configurations and 37
further runs. Walk from the first configuration to the current one **one change
at a time** — gate re-initialisation, initial bias, float32 halting — with
enough runs per step to tell a regression from a lucky draw. *Failure
condition:* if every step recovers 1.2985 at a comparable rate, those two cells
were draws and there is no regression to find.

**2. Establish the gate's success rate over runs.** ≥20 runs on the chosen
configuration, reported as `k/n` with an interval. 3/10 cannot go in a document.
Candidate levers once the baseline is known: when the penalty switches on and
how it ramps, the gate's initial bias, PonderNet-style halting (every step
receives gradient), and a curriculum on the demand mixture.

**3. Order sweep on Boxes:** DeltaProduct at `n_h ∈ {1, 2, 3}`, three seeds
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
