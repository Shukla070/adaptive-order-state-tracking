# The n_h=4 failure on S₅ alphabets — what the 80k run showed

**Date:** 12 September 2026

## The run

`S5_c5_t1`, n_h=4, **80 000 steps** (4× the original budget):

```
step   5000  loss 2.3150      step  45000  loss 2.1482
step  10000  loss 2.1965      step  50000  loss 2.1665
step  15000  loss 2.1880      step  55000  loss 2.0705
step  20000  loss 1.9282      step  60000  loss 2.0661
step  25000  loss 2.0502      step  65000  loss 1.9011
step  30000  loss 2.1883      step  70000  loss 2.0285
step  35000  loss 1.8362      step  75000  loss 1.9849
step  40000  loss 2.0377      step  80000  loss 1.9627

-> token 0.4721   seq 0.0265
```

**I was wrong at 20k.** I read `2.17 → 2.16 → 2.12 → 1.83` as a descent in
progress and said it "was on its way". It was not. Over 60 000 further steps the
loss oscillates between 1.83 and 2.19 with no trend, and accuracy went slightly
*down* (0.4851 → 0.4721). Four noisy points are not a trend, and I should not
have called one.

## What the failure actually tracks

Collecting every n_h=4 result we have, against the fraction of tokens whose
demand is D=4:

| run | group | frac D=4 | n_h=4 token acc | trains? |
|---|---|---|---|---|
| sweep p=1.00 | A₅ | 0.000 | 0.9995 | **yes** |
| `A5_c5` | A₅ | 0.000 | 0.9983 | **yes** |
| `A5_c5_dt` | A₅ | 0.000 | 0.0429 | no |
| `A5_c5_3c` | A₅ | 0.000 | 0.9993 | **yes** |
| sweep p=0.05 | S₅ | 0.050 | 0.9999 | **yes** |
| sweep p=0.10 | S₅ | 0.100 | 0.9999 | **yes** |
| sweep p=0.25 | S₅ | 0.250 | 0.9997 | **yes** |
| sweep p=0.50 | S₅ | 0.500 | 0.7070 | no |
| `S5_c5_t` | S₅ | 0.706 | 0.0706 | no |
| `S5_c5_t1` | S₅ | 0.960 | 0.4721 | no |

> **frac(D=4) ≤ 0.25 → 6 of 7 train. frac(D=4) ≥ 0.50 → 0 of 3 train.**

A sharp threshold between 25% and 50%. The one exception below the line is
`A5_c5_dt`, the plateau failure already recorded.

### This is independent support for the demand model

The table is sorted by **D**, not ℓ. Sort it by ℓ instead and it falls apart:
under ℓ, the p=1.00 run is *100% hard tokens* and should be the single worst
case in the set. It trains at **0.9995**. Under D it is 0% hard, and sits
exactly where the pattern puts it.

So D predicts **trainability**, a phenomenon it was not built for and was not
fitted to. That is a second, independent line of evidence, and it is worth more
than the cell we failed to get.

## What we lost, and what survives

**Lost:** the clean capacity threshold on `S5_c5_t1`. We cannot show "fails at
3, solves at 4" there, because at 96% high-demand tokens n_h=4 does not train
either. A skeptic can still say n_h=3's failure on that alphabet is optimisation
rather than capacity, and this run does not refute them.

**Survives — and it was already in the sweep.** The capacity threshold *is*
demonstrated, at a mixture where optimisation works:

| alphabet | group | D of a five-cycle | n_h=3 | n_h=4 |
|---|---|---|---|---|
| 25% five-cycles + transpositions (p=0.25) | S₅ | 4 | **0.5516** | **0.9997** |
| 100% five-cycles (p=1.00) | A₅ | 2 | **0.9994** | 0.9995 |

Both train at n_h=4. n_h=3 splits them cleanly. And the mixture difference runs
the *wrong way* for the skeptic: the S₅ row has **far fewer** five-cycles — the
easier mixture by any per-token account — and still fails at n_h=3, while the
A₅ row is 100% five-cycles and succeeds.

So the headline stands, with a footnote: the sharpest single-token contrast
(`A5_c5` vs `S5_c5_t1`, 0.9991 vs 0.1094) sits in a regime where n_h=4 also
fails, so it demonstrates the *phenomenon* but not cleanly the *mechanism*. The
mechanism is demonstrated at p=0.25 instead.

## Next

**1. Is the high-D-rate cliff an optimiser problem?** The oscillation between
1.83 and 2.19 is the signature of a step size too large to settle, not of
missing capacity. `--warmup` and `--cosine` are now in the script.

```bash
python work/exp_group_closure.py --arms S5_c5_t1 --orders 3,4 \
    --steps 40000 --warmup 2000 --cosine \
    --out results/group_closure_s5t1_cosine.json
```

≈40 min for both orders. If n_h=4 solves and n_h=3 still does not, we recover
the clean threshold on the single-token contrast and the whole argument closes.
If neither solves, the cliff is not an lr problem and we report the p=0.25
demonstration as the primary evidence instead.

**2. Either way, the cliff is a finding.** "Trainability collapses when more than
about a third of tokens carry maximum demand" is a statement about optimisation
that nobody has made, it is predicted by D and not by ℓ, and it deserves its own
figure — a sweep of frac(D=4) from 0 to 1 at fixed n_h=4.
