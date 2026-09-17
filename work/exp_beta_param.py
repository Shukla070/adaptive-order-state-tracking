"""
BETA PARAMETRISATION -- trying to FIX the learnability failures, not report them.

THE SEVEN FAILURES
------------------
Every one of these has provably sufficient capacity and did not train:

    sweep p=1.0   fixed2   0.1207     D=2, n_h=2
    A5_c5         n_h=2    0.1259     D=2, n_h=2
    A5_c5_dt      n_h=2    0.0443     D=2, n_h=2
    A5_c5_dt      n_h=4    0.0429     D=2, n_h=4  (its own n_h=3 got 0.9994)
    S5_c5_t1      n_h=4    0.4721     D=4, n_h=4, at 80 000 steps
    S5_c5_t       n_h=4    0.0706     D=4, n_h=4
    sweep p=0.5   fixed4   0.7070     D=4, n_h=4

These are not seven separate findings to write up. They are one problem, and
solving it is worth more than describing it.

THE HYPOTHESIS
--------------
Upstream computes beta = 2*sigmoid(W h) with bias=False. So:

  * beta lives in the OPEN interval (0,2) and can never ATTAIN 2 -- but a
    permutation needs beta exactly 2, a reflection. The layer can approach the
    solution and never reach it.

  * d beta / dx = 2*sigma*(1-sigma) -> 0 as beta -> 2. The gradient vanishes
    exactly at the value the model must reach:

        beta 1.00 -> grad 0.500     beta 1.99  -> grad 0.010
        beta 1.90 -> grad 0.095     beta 1.999 -> grad 0.001

  * and with bias=False, beta STARTS at sigmoid(0)*2 = 1.0 -- a projection,
    the one setting that destroys information, and the point of MAXIMUM
    gradient pulling away from where it needs to go.

The optimiser therefore starts at the worst value and must climb a
vanishing-gradient hill to reach the right one. If that is the mechanism, it is
fixable, and fixing it is a change to the published architecture rather than a
recombination of existing parts.

WHAT IS TRIED
-------------
  sigmoid:0    upstream. the control.
  sigmoid:4    same curve, but beta STARTS at 2*sigmoid(4) = 1.96, already a
               near-reflection. Tests "is it the starting point?" -- costs
               nothing to try and needs no new mathematics.
  clamp:0      beta = 2*clamp(0.25x + 0.5, 0, 1). ATTAINS 0 and 2 exactly,
               constant gradient 0.25 inside the range, no saturation.
               Tests "is it the vanishing gradient?"
  clamp:2      both at once: attainable endpoints AND starts at a reflection.
  ste:0        forward sigmoid, backward linear. Keeps the smooth forward map
               but removes the dead gradient. Separates the two effects.

Each also reports the beta distribution after training, which is the direct
diagnostic: a run that solved the task should show beta piled up near 2, and a
run that failed should show it stuck near 1.

FALSIFICATION
-------------
If every mode fails on a cell, the beta parametrisation is not the cause and we
move to the next candidate (curriculum on the demand mixture; orthogonality
regularisation; growing K by inserting a beta=0 factor, which our constructive
proof says is exactly how a K-factor solution embeds in K+1). Say so plainly
rather than reaching for a finding.

RUN
---
    python work/patch_beta_mode.py            # once, first
    python work/exp_beta_param.py             # fast diagnostic, ~30 min
    python work/exp_beta_param.py --cells A5_c5:2,A5_c5_dt:4,S5_c5_t1:4 \
        --modes sigmoid:0,sigmoid:4,clamp:0,clamp:2,ste:0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

DEV = "cuda"

# Numbers these cells produced in exp_group_closure.py under upstream beta.
#
# HISTORICAL ONLY -- a different script, a different process, and (because the
# bf16 kernels are non-deterministic and outcomes are bimodal) not necessarily
# the same draw. They are printed for context and MUST NOT be used to decide
# whether a mode fixed anything. The verdict uses the sigmoid(0) control run
# inside THIS invocation. See the attribution check in summarise().
HISTORICAL = {
    ("A5_c5", 2): 0.1259,
    ("A5_c5_dt", 2): 0.0443,
    ("A5_c5_dt", 4): 0.0429,
    ("A5_c5_3c", 2): 0.9510,
    ("S5_c5_t1", 4): 0.4721,
    ("S5_c5_t", 4): 0.0706,
}


def set_beta_mode(model, mode: str, bias: float) -> int:
    n = 0
    for m in model.modules():
        if hasattr(m, "beta_mode") and hasattr(m, "beta_bias"):
            m.beta_mode = mode
            with torch.no_grad():
                m.beta_bias.fill_(float(bias))
            n += 1
    if n == 0:
        raise SystemExit(
            "No layer exposes beta_mode/beta_bias.\n"
            "Run:  python work/patch_beta_mode.py")
    return n


class BetaProbe:
    """Capture the beta values the layer actually produces, and |det|.

    Reading weights is not enough: beta is a nonlinearity of a projection plus a
    bias, and what matters is where the DISTRIBUTION landed. det is the second
    half of the story -- det(I - beta k k^T) = 1 - beta, so a product of K
    factors has |det| = prod|1 - beta_i|. Upstream beta lives in the OPEN
    interval (0,2), so |det| < 1 strictly and the state contracts every step.
    A permutation has |det| = 1 exactly. Surviving 128 steps at n_h=4 needs
    beta within ~1e-3 of 2 -- precisely where the sigmoid gradient is ~1e-3.
    """

    def __init__(self, model):
        self.handles = []
        self.per_factor = {}
        for m in model.modules():
            if hasattr(m, "beta_mode") and hasattr(m, "b_projs"):
                for i, proj in enumerate(m.b_projs):
                    self.handles.append(
                        proj.register_forward_hook(self._make(m, i)))

    def _make(self, layer, i):
        def hook(_mod, _inp, out):
            raw = out.detach().float() + layer.beta_bias[i].detach().float()
            if layer.beta_mode in ("clamp", "ste"):
                b = (raw * 0.25 + 0.5).clamp(0.0, 1.0)
            else:
                b = raw.sigmoid()
            self.per_factor.setdefault(i, []).append((b * 2.0).cpu())
        return hook

    def stats(self):
        if not self.per_factor:
            return {}
        betas = {i: torch.cat(v) for i, v in self.per_factor.items()}
        v = torch.cat([x.flatten() for x in betas.values()])
        ad = abs_det_from_betas(betas)
        return {"mean": v.mean().item(), "median": v.median().item(),
                "max": v.max().item(),
                "frac_above_1_9": (v > 1.9).float().mean().item(),
                "abs_det_mean": ad.mean().item(),
                "abs_det_median": ad.median().item(),
                "abs_det_max": ad.max().item(),
                "frac_det_above_0_99": (ad > 0.99).float().mean().item()}

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def abs_det_from_betas(betas: dict) -> "torch.Tensor":
    """|det| of the product of Householder factors, elementwise.

    det(I - b k k^T) = 1 - b, so a product of K factors has
    |det| = prod_i |1 - b_i|. A permutation has |det| = 1 exactly.
    Pure function of the betas so --selftest can check it against
    hand-computed values.
    """
    det = None
    for i in sorted(betas):
        f = (1.0 - betas[i])
        det = f if det is None else det * f
    return det.abs().flatten()


def parity_counts(y, pred, par) -> dict:
    """Raw counts for the parity split of ONE batch. Pure; testable.

    Split is by the parity of the TARGET, not the prediction.

    OUT-OF-RANGE PREDICTIONS. The parity table has one entry per group element
    (120 for S5) but the model's vocabulary is larger -- it also contains
    special tokens. A model sitting on the plateau can argmax onto one of those,
    and indexing `par[pred]` with it is an out-of-bounds read. On CUDA that is a
    device-side assert which kills the process and, with it, every run still
    queued behind it. This cost 7 runs of the escape census on 13 September.

    Such a prediction is wrong by construction, since every target is a group
    element, so `eq` is already False for it. It only needs a safe parity value
    and a counter, which is reported so a model that does this a lot is visible
    rather than silently clamped.
    """
    n_par = int(par.shape[0])
    oob = pred >= n_par
    pred_safe = pred.clamp(max=n_par - 1)
    eq = pred == y
    ty, tp = par[y], par[pred_safe]
    m_even = ty == 0
    return dict(
        ev_c=int(eq[m_even].sum()), ev_n=int(m_even.sum()),
        od_c=int(eq[~m_even].sum()), od_n=int((~m_even).sum()),
        pred_even=int((tp == 0).sum()), pred_n=int(tp.numel()),
        pred_oob=int(oob.sum()),
    )


def parity_report(c: dict) -> dict:
    """Turn accumulated counts into accuracies.

    acc_odd is None -- NOT 0.0 -- when the test set contains no odd targets.

    This distinction is the whole point of this function. Every A5 alphabet
    (A5_c5, A5_c5_dt, A5_c5_3c) generates the EVEN half of S5, so every target
    is even and the odd set is empty. The earlier version divided by
    max(1, od_n), printed 0.0000, and then reported a "parity wall" on a cell
    where living in A5 is the correct and complete answer. An empty set is not
    a failed set.
    """
    ev_n, od_n = c["ev_n"], c["od_n"]
    return dict(
        acc_even=(c["ev_c"] / ev_n) if ev_n else None,
        acc_odd=(c["od_c"] / od_n) if od_n else None,
        frac_pred_even=(c["pred_even"] / c["pred_n"]) if c["pred_n"] else None,
        n_even=ev_n, n_odd=od_n, pred_oob=c.get("pred_oob", 0),
    )


def parity_verdict(r: dict) -> str:
    """One line of interpretation, or '' when the data cannot support one."""
    if r["n_odd"] == 0:
        return ("no odd targets in this test set -- this alphabet generates "
                "A5, so the odd split is undefined here, not failed")
    if r["acc_even"] is None:
        return ""
    if r["acc_even"] - r["acc_odd"] > 0.3:
        return ("PARITY WALL: right on the even half, wrong on the odd one, "
                "with predictions almost all even")
    return ""


# parity of every element of S5, for the coset diagnostic
def _parity_table():
    import itertools
    out = []
    for pm in itertools.permutations(range(5)):
        seen = [False] * 5
        c = 0
        for i in range(5):
            if not seen[i]:
                c += 1
                j = i
                while not seen[j]:
                    seen[j] = True
                    j = pm[j]
        out.append((5 - c) % 2)
    return torch.tensor(out, dtype=torch.long)


def selftest() -> bool:
    """Known-answer tests for the two derived statistics this script reports.

    Both are written to FAIL on the bugs that were actually present:
    T2 fails if an empty odd set is reported as accuracy 0, T3 fails if the
    parity-wall message fires on an A5 cell.
    """
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and cond
        print(f"  {'PASS' if cond else 'FAIL'}  {name}"
              + (f"   {detail}" if detail and not cond else ""))

    # T1 -- the parity table, against an independent method (inversion count).
    import itertools
    par = _parity_table()
    bad = 0
    for idx, pm in enumerate(itertools.permutations(range(5))):
        inv = sum(1 for i in range(5) for j in range(i + 1, 5) if pm[i] > pm[j])
        if int(par[idx]) != inv % 2:
            bad += 1
    check("T1  parity table matches inversion parity on all 120", bad == 0,
          f"{bad} disagreements")

    # T2 -- an all-even test set (every A5 alphabet). The odd split must be
    #       undefined, NOT zero. This is the bug that shipped.
    even_ids = [i for i in range(120) if int(par[i]) == 0][:8]
    y = torch.tensor(even_ids)
    c = parity_counts(y, y.clone(), par)           # perfect predictions
    r = parity_report(c)
    check("T2  empty odd set reports None, not 0.0", r["acc_odd"] is None,
          f"got {r['acc_odd']!r}")
    check("T2b n_odd is 0 and acc_even is 1.0",
          r["n_odd"] == 0 and r["acc_even"] == 1.0,
          f"n_odd={r['n_odd']} acc_even={r['acc_even']}")

    # T3 -- and no wall may be declared from it.
    check("T3  no parity-wall verdict on an all-even set",
          "PARITY WALL" not in parity_verdict(r), parity_verdict(r))

    # T4 -- a REAL wall: mixed targets, right on even, wrong on odd.
    odd_ids = [i for i in range(120) if int(par[i]) == 1][:8]
    y = torch.tensor(even_ids + odd_ids)
    pred = torch.tensor(even_ids + even_ids)       # every odd target missed
    r = parity_report(parity_counts(y, pred, par))
    check("T4  real wall detected", "PARITY WALL" in parity_verdict(r))
    check("T4b acc_even 1.0, acc_odd 0.0",
          r["acc_even"] == 1.0 and r["acc_odd"] == 0.0,
          f"{r['acc_even']} / {r['acc_odd']}")

    # T5 -- mixed targets, all correct: no wall.
    r = parity_report(parity_counts(y, y.clone(), par))
    check("T5  no wall when the model is right on both halves",
          parity_verdict(r) == "")

    # T5b -- a prediction OUTSIDE the group vocabulary must not index the
    #        parity table out of bounds. On CUDA that is a device-side assert
    #        that kills the process and every run queued behind it.
    y = torch.tensor(even_ids[:4])
    pred = torch.tensor([even_ids[0], 120, 126, even_ids[3]])   # 120,126 = specials
    try:
        c = parity_counts(y, pred, par)
        r = parity_report(c)
        ok_oob = (r["pred_oob"] == 2 and r["acc_even"] == 0.5)
    except Exception as e:
        ok_oob = False
        r = {"pred_oob": f"raised {type(e).__name__}"}
    check("T5b out-of-vocab prediction is counted, not an OOB read", ok_oob,
          f"pred_oob={r.get('pred_oob')!r}")

    # T6 -- |det| arithmetic against hand-computed values.
    #       b=2 -> |1-b|=1 (reflection, volume preserving)
    #       b=1 -> 0       (projection, collapses the direction)
    #       b=0 -> 1       (identity)
    betas = {0: torch.tensor([2.0, 1.0, 0.0, 2.0]),
             1: torch.tensor([2.0, 1.0, 2.0, 0.0])}
    want = torch.tensor([1.0, 0.0, 1.0, 1.0])
    got = abs_det_from_betas(betas)
    check("T6  |det| = prod|1-b| on known values",
          torch.allclose(got, want, atol=1e-6), f"{got.tolist()}")

    # T7 -- a single projection anywhere kills the determinant.
    betas = {0: torch.tensor([2.0]), 1: torch.tensor([1.0]),
             2: torch.tensor([2.0])}
    check("T7  one b=1 factor zeroes |det|",
          float(abs_det_from_betas(betas)[0]) == 0.0)

    print("\n  " + ("all self-tests passed" if ok else "SELF-TEST FAILED"))
    return ok


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-tailed Fisher exact p for [[a,b],[c,d]]. No scipy needed.

    Used instead of comparing medians. With a bimodal outcome and an even
    number of runs, the median jumps to whichever mode holds the majority, so
    2/4 escapes vs 1/4 escapes produces medians of 0.98 and 0.12 -- an apparent
    8x effect from a difference with p = 1.0. The escape fraction with a
    significance test is the honest comparison.
    """
    from math import comb
    n = a + b + c + d
    if n == 0:
        return 1.0

    def prob(a_, b_, c_, d_):
        return comb(a_ + b_, a_) * comb(c_ + d_, c_) / comb(n, a_ + c_)

    obs, tot = prob(a, b, c, d), 0.0
    for i in range(0, min(a + b, a + c) + 1):
        j, k, l = a + b - i, a + c - i, d - (a - i)
        if j < 0 or k < 0 or l < 0:
            continue
        p = prob(i, j, k, l)
        if p <= obs + 1e-12:
            tot += p
    return min(1.0, tot)


def summarise(rows, args) -> None:
    """Print the table and the verdict.

    The verdict is computed against the sigmoid(0) control measured in THIS
    run. The historical numbers are shown but never decide anything -- an
    earlier version compared a clamp arm against a number from a different
    script and printed FIXED for a cell whose in-run control had already
    scored 0.96.
    """
    def med(vals):
        s = sorted(vals)
        return s[len(s) // 2]

    # group by (arm, n_h, mode, bias) across seeds
    groups = {}
    for r in rows:
        groups.setdefault((r["arm"], r["n_h"], r["mode"], r["bias"]),
                          []).append(r)

    # Outcomes on these cells are BISTABLE: a run lands near-solved or near
    # chance, rarely between, and bf16 non-determinism alone decides which --
    # the same seed has produced 0.9598 and 0.1177 on A5_c5:2. So the summary
    # statistic is the ESCAPE FRACTION, never a mean over runs.
    ESCAPE = 0.5

    print("=" * 78)
    print(f"{'cell':<16} {'mode':<12} {'escape':>8} {'median':>7} "
          f"{'best':>7} {'|det| max':>9} {'hist':>7}")
    print("-" * 78)
    for (arm, n_h, mode, bias), rs in groups.items():
        toks = [r["token_acc"] for r in rs]
        n_esc = sum(1 for t in toks if t > ESCAPE)
        h = HISTORICAL.get((arm, n_h))
        print(f"{arm + ':' + str(n_h):<16} "
              f"{mode + '(' + format(bias, 'g') + ')':<12} "
              f"{f'{n_esc}/{len(toks)}':>8} "
              f"{med(toks):>7.4f} {max(toks):>7.4f} "
              f"{max(r['beta']['abs_det_max'] for r in rs):>9.4f} "
              f"{(f'{h:.4f}' if h else 'n/a'):>7}")
    print("-" * 78)
    print("  escape = runs reaching token acc > 0.5. Report this, not a mean:")
    print("  a mean over a bimodal distribution describes no actual run.")
    print()

    # An escape fraction is only meaningful if the outcomes ARE bimodal. Some
    # cells instead grind upward continuously and land all over the middle --
    # there a 0.5 cutoff invents a dichotomy that is not in the data, and the
    # "escape fraction" is an artifact of where the threshold was put.
    for (arm, n_h, mode, bias), rs in groups.items():
        toks = sorted(r["token_acc"] for r in rs)
        mid = [t for t in toks if 0.15 < t < 0.85]
        print(f"    {arm}:{n_h} {mode}({bias:g}): "
              + " ".join(f"{t:.4f}" for t in toks))
        if len(toks) >= 4 and len(mid) > len(toks) / 2:
            print(f"      !! NOT BIMODAL -- {len(mid)} of {len(toks)} runs land in the")
            print( "         middle band (0.15-0.85), so the escape fraction above is an")
            print( "         artifact of the 0.5 cutoff, NOT a property of this cell.")
            print( "         This cell grinds upward instead of jumping. Report the")
            print( "         distribution, check whether the loss is still descending at")
            print( "         the step budget, and do not call any run a 'failure'.")
    print("=" * 78)

    n_seeds = len({r["seed"] for r in rows})
    if n_seeds < 3:
        print(f"  NOTE: {n_seeds} seed(s). Outcomes on these cells are bimodal,")
        print("  so a single seed cannot distinguish a fix from a lucky draw.")
        print("  Re-run with --seeds 666,667,668 before believing any verdict.")
        print()

    # attribution check: does the in-run control disagree with history?
    for (arm, n_h, mode, bias), rs in groups.items():
        if not (mode == "sigmoid" and bias == 0):
            continue
        ctrl = med([r["token_acc"] for r in rs])
        h = HISTORICAL.get((arm, n_h))
        if h is not None and h < 0.9 <= ctrl:
            print("  !! ATTRIBUTION WARNING " + "!" * 52)
            print(f"  {arm}:n_h={n_h} is recorded historically at {h:.4f}, but the")
            print(f"  UNMODIFIED sigmoid(0) control in this run scored {ctrl:.4f}.")
            print("  Same parametrisation, so the historical number is not a")
            print("  reproducible baseline for this cell. Do not credit any mode")
            print("  with fixing it until that gap is explained -- the cell may")
            print("  simply be bimodal, in which case it was never a failure.")
            print("  " + "!" * 74)
            print()

    # verdict, against the in-run control only, on escape fraction
    def escapes(rs):
        t = [r["token_acc"] for r in rs]
        return sum(1 for x in t if x > ESCAPE), len(t)

    controls = {(a, n): escapes(rs) for (a, n, m, b), rs in groups.items()
                if m == "sigmoid" and b == 0}
    if not controls:
        print("  NO VERDICT: this run contains no sigmoid(0) control arm, so")
        print("  there is nothing to compare against. Add sigmoid:0 to --modes.")
        return

    fixed, inconclusive = [], []
    for (arm, n_h, mode, bias), rs in groups.items():
        if mode == "sigmoid" and bias == 0:
            continue
        if (arm, n_h) not in controls:
            continue
        ce, cn = controls[(arm, n_h)]
        ae, an = escapes(rs)
        p = fisher_exact_2x2(ae, an - ae, ce, cn - ce)
        entry = (arm, n_h, mode, bias, ae, an, ce, cn, p)
        (fixed if (ae / an > ce / cn and p < 0.05) else inconclusive).append(entry)

    for arm, n_h, mode, bias, ae, an, ce, cn, p in fixed:
        print(f"  FIXED — {mode}(bias {bias:g}) escapes more often than the "
              f"control on {arm}:n_h={n_h}")
        print(f"    {ae}/{an} vs control {ce}/{cn}   Fisher p = {p:.4f}")
        print("  Next: check it does not regress the cells that already worked,")
        print("  then re-run the group-closure grid with it.")
    for arm, n_h, mode, bias, ae, an, ce, cn, p in inconclusive:
        print(f"  INCONCLUSIVE — {mode}(bias {bias:g}) on {arm}:n_h={n_h}: "
              f"{ae}/{an} escapes vs control {ce}/{cn}, Fisher p = {p:.4f}")
        print(f"    Not enough runs to tell these apart. To detect a jump from")
        print(f"    25% to 75% escape at p<0.05 with 80% power needs about 20")
        print(f"    runs PER ARM; you have {an} and {cn}.")
    if not fixed:
        print()
        print("  NOT ESTABLISHED against the in-run control.")
        print()
        print("  Two claims must be kept apart here:")
        print("   (a) ATTAINABILITY. clamp reaches beta = 2.0000 and |det| =")
        print("       1.0000 exactly; sigmoid tops out near 0.998 and never")
        print("       reaches 1. That is the open-vs-closed interval showing up")
        print("       numerically, and it holds in every run.")
        print("   (b) LEARNING. Attainability does NOT imply the task is")
        print("       learned. Runs have reached |det| = 1.0000 and still scored")
        print("       at chance, while a run that solved it sat at 0.69. Do not")
        print("       cite |det| as evidence the cell trains.")
        print("  Next candidates, in order of cost:")
        print("    1. curriculum on the demand mixture (low frac(D=max) first --")
        print("       the trainability table says that regime trains reliably)")
        print("    2. grow K by inserting a beta=0 factor into a trained K-1")
        print("       model -- our construction proof says that embedding is exact")
        print("    3. orthogonality regularisation on the transition matrices")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="A5_c5:2",
                    help="arm:n_h pairs, comma separated")
    ap.add_argument("--modes", default="sigmoid:0,sigmoid:4,clamp:2",
                    help="mode:bias pairs, comma separated")
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--n_train", type=int, default=100000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_heads", type=int, default=12)
    ap.add_argument("--head_dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=666,
                    help="single run seed (ignored if --seeds is given)")
    ap.add_argument("--seeds", default=None,
                    help="comma-separated run seeds, e.g. 666,667,668. "
                         "Varies model init and batch order ONLY -- the data "
                         "is held fixed at --data_seed so the two sources of "
                         "variation are not confounded.")
    ap.add_argument("--data_seed", type=int, default=666,
                    help="seed for dataset generation; held fixed across seeds")
    ap.add_argument("--eval_seqs", type=int, default=0,
                    help="sequences to evaluate on (0 = the whole test set)")
    ap.add_argument("--probe_batches", type=int, default=3,
                    help="batches to accumulate beta statistics over")
    ap.add_argument("--log_every", type=int, default=5000)
    ap.add_argument("--out", default="results/beta_param.json")
    ap.add_argument("--selftest", action="store_true",
                    help="run known-answer tests on the diagnostics and exit")
    args = ap.parse_args()

    if args.selftest:
        print("=" * 78)
        print("SELF-TEST -- diagnostics against data whose answer is known")
        print("=" * 78)
        raise SystemExit(0 if selftest() else 1)

    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else [args.seed])

    from exp_matched_compute import build, N_GROUP, N_SPECIAL
    from exp_group_closure import ALPHABETS, make_data, closure, predicted_demand
    args.vocab = N_GROUP + N_SPECIAL
    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA")

    cells = []
    for c in args.cells.split(","):
        a, n = c.strip().split(":")
        cells.append((a.strip(), int(n)))
    modes = []
    for m in args.modes.split(","):
        mm, b = m.strip().split(":")
        modes.append((mm.strip(), float(b)))

    print("=" * 78)
    print("BETA PARAMETRISATION -- can we make the failed cells train?")
    print("=" * 78)
    print("  hypothesis: beta = 2*sigmoid(x) cannot attain 2, and its gradient")
    print("  vanishes as it approaches 2 -- the value a permutation needs.")
    print("  With bias=False it also STARTS at 1.0, the worst possible value.")
    print()
    for a, n in cells:
        base = HISTORICAL.get((a, n))
        clo = closure(ALPHABETS[a])
        D = predicted_demand(len(clo))
        grp = "A5" if len(clo) == 60 else ("S5" if len(clo) == 120 else str(len(clo)))
        print(f"  cell {a}:n_h={n}   <A> = {grp}   D={D}   "
              f"historical (other script): "
              f"{('%.4f' % base) if base else 'n/a'}")
        if grp == "A5":
            print("       note: A5 is the even half of S5, so this cell has NO")
            print("       odd targets and the parity split is undefined on it.")
    print(f"  modes: {', '.join(f'{m}(bias {b:g})' for m, b in modes)}")
    print(f"  seeds: {', '.join(str(s) for s in seeds)}  "
          f"(data fixed at seed {args.data_seed})")
    print(f"  budget: {args.steps} steps x batch {args.batch}")
    if len(seeds) < 3:
        print("  WARNING: fewer than 3 seeds. Outcomes here are bimodal; a")
        print("  single seed cannot tell a fix from a lucky draw.")
    print()

    rows = []
    for arm, n_h in cells:
        alpha = ALPHABETS[arm]
        inp, tgt = make_data(alpha, args.k, args.n_train + args.n_test,
                             args.data_seed)
        tr_x = inp[:args.n_train].to(DEV)
        tr_y = tgt[:args.n_train].to(DEV)
        te_x = inp[args.n_train:].to(DEV)
        te_y = tgt[args.n_train:].to(DEV)

        for mode, bias in modes:
          for seed in seeds:
            tag = (f"{arm}:n_h={n_h}:{mode}({bias:g})"
                   + (f":seed{seed}" if len(seeds) > 1 else ""))
            print("-" * 78)
            print(tag)
            print("-" * 78)
            model = build(n_h, False, args.n_heads, args.head_dim, args.hidden,
                          args.n_layers, args.vocab, seed)
            set_beta_mode(model, mode, bias)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
            g = torch.Generator(device=DEV).manual_seed(seed)
            t0 = time.perf_counter()
            # Keep the trace in the JSON, not just the log. Escape from the
            # plateau is abrupt and its TIMING varies: one run escaped before
            # step 5000, another was still at loss 3.72 at 5000 and at 0.008 by
            # 10000. Without the trace there is no way to tell a run that never
            # escaped from one that escaped late.
            trace = []
            for step in range(1, args.steps + 1):
                i = torch.randint(0, tr_x.shape[0], (args.batch,),
                                  device=DEV, generator=g)
                loss = F.cross_entropy(
                    model(input_ids=tr_x[i]).logits.float().flatten(0, 1),
                    tr_y[i].flatten())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if step % args.log_every == 0 or step == 1:
                    trace.append((step, round(loss.item(), 4)))
                    print(f"      step {step:>6}  loss {loss.item():.4f}")
            dt = time.perf_counter() - t0

            model.eval()
            probe = BetaProbe(model)
            par = _parity_table().to(DEV)
            tc = tt = sc = nseq = 0
            pc = dict(ev_c=0, ev_n=0, od_c=0, od_n=0, pred_even=0, pred_n=0,
                      pred_oob=0)
            # Accuracy is measured over the WHOLE test set. The previous
            # version broke out of this loop at i >= 256, so every headline
            # number came from 384 sequences while every other script in the
            # project reports over 2000 -- not comparable, and noisier.
            # Only the beta probe is capped, and only because it accumulates.
            limit = args.eval_seqs or te_x.shape[0]
            with torch.no_grad():
                for bi, i in enumerate(range(0, min(limit, te_x.shape[0]), 128)):
                    if bi == args.probe_batches:
                        probe.close()      # stop accumulating, keep evaluating
                    y = te_y[i:i + 128]
                    pred = model(input_ids=te_x[i:i + 128]).logits.float().argmax(-1)
                    eq = pred == y
                    tc += eq.sum().item()
                    tt += eq.numel()
                    sc += eq.all(-1).sum().item()
                    nseq += y.shape[0]
                    for k_, v_ in parity_counts(y, pred, par).items():
                        pc[k_] += v_
            bs = probe.stats()
            probe.close()

            tok, seq = tc / tt, sc / max(1, nseq)
            pr = parity_report(pc)
            acc_even, acc_odd = pr["acc_even"], pr["acc_odd"]
            frac_pred_even = pr["frac_pred_even"]
            base = HISTORICAL.get((arm, n_h))
            delta = (f"  (historical {base:.4f})" if base else "")
            print(f"    -> token {tok:.4f}   seq {seq:.4f}   "
                  f"({dt:.0f}s, {nseq} seqs){delta}")
            print(f"       beta : mean {bs['mean']:.3f}  median {bs['median']:.3f}"
                  f"  max {bs['max']:.4f}  frac>1.9 {bs['frac_above_1_9']:.3f}")
            print(f"       |det|: mean {bs['abs_det_mean']:.4f}  "
                  f"median {bs['abs_det_median']:.4f}  "
                  f"max {bs['abs_det_max']:.4f}  "
                  f"frac>0.99 {bs['frac_det_above_0_99']:.3f}   (1.0 = a permutation)")
            fe = f"{acc_even:.4f}" if acc_even is not None else "n/a"
            fo = f"{acc_odd:.4f}" if acc_odd is not None else "n/a"
            print(f"       parity: EVEN targets {fe} (n={pr['n_even']})   "
                  f"ODD targets {fo} (n={pr['n_odd']})   "
                  f"predictions even {frac_pred_even:.3f}")
            v = parity_verdict(pr)
            if v:
                print(f"               -> {v}")
            print()
            rows.append(dict(arm=arm, n_h=n_h, mode=mode, bias=bias, seed=seed,
                             token_acc=tok, seq_acc=seq, train_s=dt,
                             n_eval_seqs=nseq, historical=base, beta=bs,
                             loss_trace=trace,
                             acc_even=acc_even, acc_odd=acc_odd,
                             n_even=pr["n_even"], n_odd=pr["n_odd"],
                             frac_pred_even=frac_pred_even))
            del model
            torch.cuda.empty_cache()

            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({"config": vars(args), "results": rows},
                                      indent=2, default=str))
        del tr_x, tr_y, te_x, te_y
        torch.cuda.empty_cache()

    summarise(rows, args)
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
