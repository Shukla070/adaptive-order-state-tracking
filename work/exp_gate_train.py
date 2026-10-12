"""
LEARNED GATE, TRAINED END-TO-END.  Objective O4.

Everything adaptive in this project so far has come from an ORACLE -- a control
condition handed the correct per-token order from ground truth. This file is the
first experiment in which the halting gate receives no supervision on the
allocation and has to infer it from the task.

WHAT IS MEASURED
----------------
Three arms, all built and trained inside ONE invocation so the comparison needs
no cross-script baseline:

    fixed<K>   fixed order K, no gate.        Accuracy reference.
    oracle     gates = L(token) from truth.   Cost/accuracy target.
    learned    the halting gate + budget loss. The thing being tested.

and two questions, which are NOT the same question:

    Q1  COST.  Does the learned gate reach oracle accuracy at a cost
        approaching E[D]?  (At p=0.10, E[D] = 1.30 against a fixed cost of 4.)

    Q2  ALLOCATION.  Does it spend MORE on tokens that genuinely demand more?
        A gate can hit the right average by closing uniformly, which would be
        worthless. So the report splits mean n_t by the token's true demand and
        reports mean|n_t - D| and the fraction of tokens UNDER-allocated.
        Under-allocation is the failure that breaks correctness; over-allocation
        only wastes compute. They are reported separately, never as |error|.

WHAT THIS DOES NOT ESTABLISH
----------------------------
Nothing about wall-clock. Path A still expands all K factors and multiplies the
disabled ones by zero, so a lower n_t costs exactly the same time. Cost here is
required work per token. Path B is a separate work item.

PROTOCOL
--------
Gates are SOFT while training (smooth gradients) and HARD at evaluation, so the
reported cost is a true integer factor count rather than a sum of sigmoids. The
gate initialises open, so at step 0 the model is fixed-order DeltaProduct(K),
which is the configuration known to converge; the budget penalty is then ramped
in, rather than applied from step 1, so the gate cannot close before the task is
learned. `--gamma 0` is the control for the gate's mere presence: gate installed,
never penalised, should track the fixed arm.

Outcomes on these cells are bistable (PLAN Part 3c), so every configuration is
run repeatedly and reported as a distribution, never a point estimate.

RUN
---
    python work/exp_gate_train.py --selftest          # no GPU, instant

    python work/exp_gate_train.py --p 0.1 --gammas 0,0.01,0.03,0.1 \
        --seeds 1,2,3 --out results/gate_train_p10.json

    # gate-side hyperparameters, holding everything else fixed
    python work/exp_gate_train.py --p 0.1 --gammas 0.03 --seeds 1,2,3,4,5 \
        --gate_init_bias 2.0 --gate_lr 3e-3 --arms learned \
        --out results/gate_train_p10_bias2.json

WHY THE GATE HAS ITS OWN LEARNING RATE AND INIT
-----------------------------------------------
The gate's output weight is initialised to exactly zero and its bias to
`--gate_init_bias`, so at step 0 the gate is a constant: every token gets the
same order regardless of content. It can only start allocating once that weight
moves off zero, and the gradient reaching it is scaled by the sigmoid slope at
the init bias -- 0.018 at 4.0 against 0.105 at 2.0. Both flags target that
single bottleneck. Passing neither reproduces the runs from gate_fixedinit
onwards -- NOT the runs before them, which started from a different gate.

RE-RUNNING THE RUNS MADE BEFORE THE INIT REPAIR
-----------------------------------------------
gate_train_p10 and gate_bias40_more were run before build() re-initialised the
gate, so HuggingFace post_init had left it at bias 0 and small random weights:
mean n_t 0.93-0.95 at step 1 (measured in gate_bias40_more, whose header
nevertheless printed "init bias 4"). `--gate_init hf_default` rebuilds that
starting point. Two guards make a mismatch visible instead of silent:

  * the learned arm checks its step-1 mean n_t against the value --gate_init
    implies and aborts if they disagree -- the check that would have caught the
    gate_bias40_more header;
  * `--fingerprint <old results json>` compares step-1 task loss (and, where the
    old run logged them, step-1 n_t and gate |W|) with the run being
    reconstructed, per seed. A MISMATCH means the model being trained is not
    the model that produced the old number.

    # cheap: does the reconstruction start where the old runs started? (~1 min)
    python work/exp_gate_train.py --gate_init hf_default --arms oracle,learned \
        --gammas 0.03 --seeds 1,2,3 --steps 1 \
        --fingerprint results/gate_train_p10.json --out /tmp/fp_p10.json

THE LOGIT WALL (--logit_walls)
------------------------------
Rebuilding the earliest configuration (PLAN Part 3e) showed which seeds close is
decided in the first ~1 000 steps, before the budget penalty exists: the task
loss pushes every halting logit up, and once a logit passes 6.2364 the bf16
sigmoid is exactly 1.0 with exactly zero gradient, so nothing can ever close it.

`--logit_walls 5` adds  rho * mean(relu(|z| - 5)^2)  over every halting logit z,
from step 1, with rho = --logit_wall_weight. It is zero inside [-5, 5] and its
gradient, 2*rho*(|z| - 5), does not pass through the sigmoid, so it acts exactly
where the sigmoid's own gradient has vanished. It is a penalty and not a clamp
on purpose: torch.clamp has zero gradient outside its range, which would
reproduce the frozen gate, and a straight-through clamp lets z drift without
limit beyond the cap while the forward pass shows nothing.

Known side effect, and its control: Adam normalises step size, so while any
logit sits above the wall the shared output bias keeps moving down, dragging
other tokens' logits with it. With no task loss opposing it, the wall alone
lowers n_t (self-test T17d). In training the task loss opposes it -- but the
γ = 0 arm with the wall on is the control that says how much closure the wall
causes without any budget.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

DEV = "cuda"
DTYPE = torch.bfloat16
N_SPECIAL = 7
N_GROUP = 120


# ===========================================================================
# Pure helpers -- every one of these is covered by --selftest
# ===========================================================================

def budget_weight(step: int, gamma: float, warmup: int, ramp: int) -> float:
    """Effective budget weight at `step`.

    Zero for the first `warmup` steps, then linear to `gamma` over `ramp`
    steps, then constant. The gate starts open so that the model begins as
    fixed-order DeltaProduct(K); penalising from step 1 can close it before the
    task is learned, which would measure the schedule rather than the gate.
    """
    if gamma == 0.0:
        return 0.0
    if step <= warmup:
        return 0.0
    if ramp <= 0:
        return gamma
    return gamma * min(1.0, (step - warmup) / ramp)


def _pearson(a: torch.Tensor, b: torch.Tensor):
    """Pearson correlation, or None when either side has no variance."""
    a = a.float().flatten()
    b = b.float().flatten()
    if a.numel() < 2:
        return None
    va, vb = a - a.mean(), b - b.mean()
    da, db = va.norm().item(), vb.norm().item()
    if da < 1e-8 or db < 1e-8:
        return None
    return float((va @ vb).item() / (da * db))


def allocation_stats(n_t: torch.Tensor, input_ids: torch.Tensor,
                     ell_lookup: torch.Tensor, max_order: int,
                     n_group: int = N_GROUP) -> dict:
    """Compare the spent order against the true demand, token by token.

    n_t         (B, T) float -- factors the gate actually spent
    input_ids   (B, T) long
    ell_lookup  (vocab,) long -- transposition length per token id
    max_order   K; the demand is clipped to this, because K factors is all the
                model has. A 5-cycle under K=2 demands 2, not 4, and scoring it
                against 4 would report a shortfall the model cannot avoid.
    n_group     ids >= n_group are special tokens carrying no group element.
                They are EXCLUDED from every statistic here: their correct
                transition is the identity, so counting them would dilute the
                cost with free tokens.

    Over- and under-allocation are reported separately. Under-allocation is the
    one that breaks correctness; a single |error| figure hides the difference.
    """
    ell = ell_lookup.to(input_ids.device)[input_ids]
    keep = input_ids < n_group
    n_excluded = int((~keep).sum().item())
    if not bool(keep.any()):
        return dict(n_eval=0, n_special_excluded=n_excluded, mean_n_t=None,
                    by_demand={}, mae=None, under_frac=None, over_frac=None,
                    corr=None)

    spent = n_t.float()[keep]
    demand = ell.clamp(min=0, max=max_order).float()[keep]
    diff = spent - demand

    by_demand = {}
    for d in sorted(set(int(v) for v in demand.unique().tolist())):
        sel = demand == d
        by_demand[d] = dict(n=int(sel.sum().item()),
                            mean_n_t=float(spent[sel].mean().item()))

    return dict(
        n_eval=int(keep.sum().item()),
        n_special_excluded=n_excluded,
        mean_n_t=float(spent.mean().item()),
        mean_demand=float(demand.mean().item()),
        by_demand=by_demand,
        mae=float(diff.abs().mean().item()),
        under_frac=float((diff < -1e-6).float().mean().item()),
        over_frac=float((diff > 1e-6).float().mean().item()),
        corr=_pearson(spent, demand),
    )


CLOSE_FRAC = 0.95      # a run "closed the gate" if its cost fell below this * K


def is_closed(row: dict, close_frac: float = CLOSE_FRAC) -> bool:
    """Did this run's gate close at all, or is it still paying full order?"""
    c = row["alloc"]["mean_n_t"]
    return c is not None and c < close_frac * row["n_h"]


def closure_fraction(arms: list, close_frac: float = CLOSE_FRAC):
    """(closed, total) over a set of runs.

    Outcomes on these cells are bistable (PLAN Part 3c): a run either closes the
    gate or does not, and reporting the MEAN cost over such runs -- or the cost
    of the best-accuracy one -- describes a run that did not happen. Everything
    downstream of this function reports the fraction instead.
    """
    return sum(1 for r in arms if is_closed(r, close_frac)), len(arms)


def gate_verdict(learned: dict, oracle: dict, n_h: int,
                 acc_tol: float = 0.01) -> str:
    """One line on what a learned arm achieved, against the in-run oracle."""
    a_l, a_o = learned["token_acc"], oracle["token_acc"]
    c_l = learned["alloc"]["mean_n_t"]
    if c_l is None:
        return "NO DATA"
    if a_l < 0.5:
        return ("BUDGET TOO STRONG — accuracy collapsed; the gate closed before "
                "the task was learned")
    if c_l > 0.95 * n_h:
        return ("GATE DID NOT CLOSE — cost is still ~K, so the budget penalty is "
                "too weak to bite")
    if a_l >= a_o - acc_tol:
        return (f"USEFUL ALLOCATION — oracle accuracy held at cost {c_l:.3f} "
                f"against a fixed cost of {n_h}")
    return (f"ACCURACY TRADED FOR COST — {a_o - a_l:+.4f} accuracy against the "
            f"oracle, at cost {c_l:.3f}")


# ===========================================================================
# Model plumbing
# ===========================================================================

def adaptive_layers(model):
    return [m for m in model.modules() if getattr(m, "adaptive_order", False)]


def collect_n_t(model):
    """Per-layer (B, T) effective orders from the most recent forward pass."""
    out = [m.last_n_t for m in adaptive_layers(model) if m.last_n_t is not None]
    if not out:
        raise RuntimeError(
            "no layer reported last_n_t. Either the model was built with "
            "adaptive=False, or the PATCH(adaptive-order) block is missing from "
            "fla/layers/gated_deltaproduct.py")
    return out


def set_gate_hard(model, hard: bool) -> int:
    n = 0
    for m in adaptive_layers(model):
        m.gate_hard = hard
        n += 1
    return n


def gate_parameters(model):
    """(names, params) of the halting gate only.

    Matching is on the module object, not on a name substring, so a rename of
    the attribute cannot silently produce an empty list.
    """
    names, params = [], []
    gate_ids = set()
    for m in adaptive_layers(model):
        g = getattr(m, "halting_gate", None)
        if g is not None:
            gate_ids.update(id(p) for p in g.parameters())
    for name, p in model.named_parameters():
        if id(p) in gate_ids:
            names.append(name)
            params.append(p)
    return names, params


def param_groups(model, lr: float, gate_lr: float | None):
    """Optimizer groups: everything at `lr`, the gate at `gate_lr`.

    Raises when `gate_lr` is requested but no gate parameter is found. That is
    the dangerous failure: the run would complete, report a number, and have
    measured the default learning rate under a flag that claims otherwise.
    """
    if gate_lr is None:
        return [dict(params=list(model.parameters()), lr=lr)]
    names, gate_p = gate_parameters(model)
    if not gate_p:
        raise RuntimeError(
            "--gate_lr was given but this model has no halting-gate parameters. "
            "Either the arm is not adaptive, or the gate module is no longer "
            "reachable as `layer.halting_gate`.")
    gate_ids = {id(p) for p in gate_p}
    rest = [p for p in model.parameters() if id(p) not in gate_ids]
    return [dict(params=rest, lr=lr), dict(params=gate_p, lr=gate_lr)]


def gate_weight_norm(model):
    """Sum of |W| over the gate's OUTPUT-layer weight, or None if no gate.

    That weight is initialised to exactly zero, which makes the gate
    input-INDEPENDENT: every token receives the same order whatever it is. Until
    the weight moves off zero the gate cannot allocate at all, only raise or
    lower one global order. Logging it separates two failures that look
    identical in the cost column -- "the gate stayed open" from "the gate never
    became a function of its input".
    """
    tot, found = 0.0, False
    for m in adaptive_layers(model):
        g = getattr(m, "halting_gate", None)
        if g is None:
            continue
        tot += g.mlp[-1].weight.detach().float().abs().sum().item()
        found = True
    return tot if found else None


# ---------------------------------------------------------------------------
# Demand: which order each input token needs (PLAN Part 3.3)
# ---------------------------------------------------------------------------
# Token id == index into itertools.permutations(range(5)), as written by
# work/gen_heterogeneous.py, so the permutation behind every token is known.

def _s5_elements():
    import itertools
    return list(itertools.permutations(range(5)))


def _ell(p) -> int:
    """Transposition length = 5 - number of cycles."""
    seen, cycles = set(), 0
    for i in range(len(p)):
        if i in seen:
            continue
        cycles += 1
        j = i
        while j not in seen:
            seen.add(j)
            j = p[j]
    return len(p) - cycles


def _generated_group(alphabet_ids):
    """Set of element indices generated by the given elements (closure)."""
    els = _s5_elements()
    index = {p: i for i, p in enumerate(els)}
    gens = [els[i] for i in alphabet_ids]
    ident = tuple(range(5))
    seen, frontier = {ident}, [ident]
    while frontier:
        nxt = []
        for a in frontier:
            for g in gens:
                c = tuple(a[g[i]] for i in range(5))
                if c not in seen:
                    seen.add(c)
                    nxt.append(c)
        frontier = nxt
    return {index[p] for p in seen}


def demand_from_alphabet(alphabet_ids, rule: str):
    """List of length 120: demand D of every S5 element, given the input alphabet.

    rule 'ell'   transposition length -- the cost model every run before
                 8 October used.
    rule 'model' the demand model D(g) = min over faithful representations of
                 the generated group of rank(rho(g) - I) (PLAN Part 3.3):
                   generated group S5 -> D = ell (standard 4-dim rep);
                   generated group A5 -> D = 2 for every non-identity element
                                         (3-dim icosahedral rep).
                 Any other generated group raises: not implemented, never
                 guessed.
    """
    els = _s5_elements()
    ell = [_ell(p) for p in els]
    if rule == "ell":
        return ell
    if rule != "model":
        raise ValueError(f"unknown demand rule {rule!r}")
    group = _generated_group(alphabet_ids)
    if len(group) == 120:
        return ell
    if len(group) == 60 and all(ell[i] % 2 == 0 for i in group):
        return [0 if e == 0 else 2 for e in ell]
    raise ValueError(f"demand model not implemented for a generated group of "
                     f"order {len(group)}")


def alphabet_from_meta(meta: dict):
    """Input alphabet of a gen_heterogeneous.py data file: every element of
    length hard_ell if p_hard > 0, plus every element of length easy_ell if
    p_hard < 1."""
    els = _s5_elements()
    want = set()
    if meta["p_hard"] > 0:
        want.add(meta["hard_ell"])
    if meta["p_hard"] < 1:
        want.add(meta["easy_ell"])
    return [i for i, p in enumerate(els) if _ell(p) in want]


BF16_SAT_LOGIT = 6.2364   # sigmoid rounds to exactly 1.0 in bf16 above this
FP32_SAT_LOGIT = 17.3287  # and in float32 above this (check_gate_precision.py)


def gate_logits(model):
    """Pre-sigmoid halting logits each gate recorded on the last forward.

    Raises rather than returning nothing: a wall computed over an empty list is
    zero, which would make --logit_walls a silent no-op.
    """
    out = []
    for m in adaptive_layers(model):
        g = getattr(m, "halting_gate", None)
        if g is None:
            continue
        z = getattr(g, "last_logits", None)
        if z is None:
            raise RuntimeError(
                "a halting gate recorded no logits. Either no forward pass has "
                "run yet, or src/gating.py predates last_logits.")
        out.append(z)
    if not out:
        raise RuntimeError("no halting gate found; --logit_walls has nothing to act on")
    return out


def logit_wall_loss(logits, c: float) -> torch.Tensor:
    """mean over all halting logits of relu(|z| - c)^2, in float32.

    Zero while every |z| <= c. The gradient 2*(|z| - c)*sign(z)/N does not pass
    through the sigmoid, so it is non-zero exactly where the sigmoid's own
    gradient has vanished.
    """
    return torch.stack([torch.relu(z.float().abs() - c).pow(2).mean()
                        for z in logits]).mean()


def saturation_stats(model):
    """(largest halting logit, fraction of halting probabilities exactly 1.0).

    Exact equality in the dtype the sigmoid was evaluated in, so this measures
    saturation directly rather than inferring it from a threshold.
    """
    zmax, sat = [], []
    for m in adaptive_layers(model):
        g = getattr(m, "halting_gate", None)
        if g is None or g.last_logits is None:
            continue
        zmax.append(g.last_logits.detach().float().max().item())
        sat.append((g.last_lambdas == 1).float().mean().item())
    if not zmax:
        return None, None
    return max(zmax), sum(sat) / len(sat)


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for k successes in n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def is_separated(row, acc_floor: float = 0.99, soft: bool = False) -> bool:
    """The rule used everywhere since Part 3e: allocation correlates with
    demand at >= 0.99 AND the task is kept (token accuracy >= 0.99).

    soft=True judges accuracy by the soft-gate evaluation instead of the hard
    one; False when the run has no soft evaluation."""
    c = row["alloc"].get("corr")
    if soft:
        se = row.get("soft_eval")
        if not se:
            return False
        acc = se["token_acc"]
    else:
        acc = row["token_acc"]
    return c is not None and c >= 0.99 and acc >= acc_floor


def is_exact(row, acc_floor: float = 0.99, tol: float = 0.01) -> bool:
    """Every demand class receives exactly min(D, K) factors, task kept."""
    by = row["alloc"].get("by_demand") or {}
    if not by or row["token_acc"] < acc_floor:
        return False
    K = row["n_h"]
    return all(abs(v["mean_n_t"] - min(int(d), K)) < tol for d, v in by.items())


def arm_summary(rows):
    """k/n separated and k/n exact with Wilson intervals, plus the sorted costs."""
    n = len(rows)
    ks = sum(is_separated(r) for r in rows)
    ke = sum(is_exact(r) for r in rows)
    costs = sorted(r["alloc"]["mean_n_t"] for r in rows
                   if r["alloc"]["mean_n_t"] is not None)
    kss = sum(is_separated(r, soft=True) for r in rows)
    return dict(n=n, separated=ks, sep_ci=wilson(ks, n),
                exact=ke, exact_ci=wilson(ke, n), costs=costs,
                separated_soft=kss, sep_soft_ci=wilson(kss, n),
                has_soft=any(r.get("soft_eval") for r in rows))


def gates_hard_at(step: int, hard_from: int) -> bool:
    """Training uses hard (straight-through) gates from step `hard_from` on.
    0 means never: soft gates for the whole run, as in every run before 2 Oct."""
    return hard_from > 0 and step >= hard_from


@torch.no_grad()
def evaluate_gates(model, te_x, te_y, hard: bool, batch: int = 128):
    """Token and sequence accuracy, and mean n_t over all positions, with the
    gates in the requested mode.

    It sets the mode itself, so a caller cannot silently evaluate in whatever
    mode training left behind. mean_n_t_all includes special tokens, so it is
    not the allocation statistic (which excludes them); it is for comparing the
    soft and hard passes of the same model.
    """
    if set_gate_hard(model, hard) == 0:
        raise RuntimeError("evaluate_gates: model has no adaptive layer")
    model.eval()
    tc = tt = sc = ns = 0
    nsum = ncount = 0.0
    for i in range(0, te_x.shape[0], batch):
        x, y = te_x[i:i + batch], te_y[i:i + batch]
        pred = model(input_ids=x).logits.float().argmax(-1)
        eq = pred == y
        tc += eq.sum().item(); tt += eq.numel()
        sc += eq.all(-1).sum().item(); ns += y.shape[0]
        n_t = collect_n_t(model)[0].float()
        nsum += n_t.sum().item(); ncount += n_t.numel()
    return dict(token_acc=tc / tt, seq_acc=sc / max(1, ns),
                mean_n_t_all=nsum / ncount)


def expected_start_order(gate_init: str, bias: float, K: int, hard: bool = False,
                         min_order: int = 0):
    """(expected mean n_t at step 1, tolerance).

    Gate i (1-based) is the product of halting probabilities 1..i, and the first
    `min_order` of those are fixed at 1. With every probability at lam:
        g_i = 1 for i <= m,   g_i = lam ** (i - m) otherwise.
    Soft: n_t = sum g_i (3.8233 at bias 4, K 4, m 0). Hard: the number of g_i
    above 0.5 (4 at bias 4). The pre-repair start has lam ~0.5 per token, so its
    hard start is only known to lie between m and m + 1, and with m = 0 it is
    ~0 -- a layer that writes nothing -- which is refused.
    """
    m = min(min_order, K)
    if gate_init == "repaired":
        lam = 1.0 / (1.0 + math.exp(-bias))
    elif gate_init == "hf_default":
        lam = 0.5
    else:
        raise ValueError(f"unknown gate_init {gate_init!r}")
    g = [1.0 if i <= m else lam ** (i - m) for i in range(1, K + 1)]
    if not hard:
        return sum(g), (5e-3 if gate_init == "repaired" else 0.02)
    if gate_init == "repaired":
        return float(sum(x > 0.5 for x in g)), 1e-6
    if m == 0:
        raise ValueError("hard gates from step 1 with the pre-repair start "
                         "begin at order ~0; use --gate_init repaired or "
                         "--min_order 1")
    return m + 0.5, 0.5


def fingerprint_check(ref_rows, mode, seed, task, n_t=None, w=None,
                      tol_task=1.5e-4, tol_nt=2e-3, tol_w_rel=0.02):
    """Compare a run's step-1 values with the run it is meant to reproduce.

    Step-1 loss is fixed by the initial weights and the first batch, both
    seeded, so a faithful reconstruction matches it to the 4 decimals stored.
    Identical configurations in this project agree at step 1 to <= 1e-4; the
    open-gate and pre-repair models differ by 3e-4 to 1.3e-3 on seeds 1-3.
    n_t and |W| are compared only where the old run logged them (gate_train_p10
    printed n_t as 0.000 during warmup, so it is skipped there).
    """
    cand = [r for r in ref_rows if r.get("mode") == mode and r.get("seed") == seed
            and r.get("loss_trace") and r["loss_trace"][0][0] == 1]
    if not cand:
        return dict(verdict="NO REFERENCE", detail=f"no {mode} seed {seed} in reference")
    ref = cand[0]["loss_trace"][0]
    checks = [("task", task, ref[1], abs(task - ref[1]) <= tol_task)]
    if mode == "learned" and n_t is not None and len(ref) > 2 and ref[2]:
        checks.append(("n_t", n_t, ref[2], abs(n_t - ref[2]) <= tol_nt))
    if (mode == "learned" and w is not None and len(ref) > 3
            and ref[3] is not None):
        checks.append(("|W|", w, ref[3],
                       abs(w - ref[3]) <= tol_w_rel * max(abs(ref[3]), 1e-12)))
    ok = all(c[3] for c in checks)
    detail = "  ".join(f"{n} {got:.4f} vs {want:.4f}{'' if good else ' <<'}"
                       for n, got, want, good in checks)
    return dict(verdict="MATCH" if ok else "MISMATCH", detail=detail)


# ===========================================================================
# Self-test -- every check is verified to FAIL against a wrong implementation
# ===========================================================================

def selftest() -> bool:
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}"
              + (f"   {detail}" if detail and not cond else ""))

    # ---- budget ramp -----------------------------------------------------
    g, w, r = 0.1, 1000, 2000
    pts = [(1, 0.0), (1000, 0.0), (1001, g * 0.0005), (2000, g * 0.5),
           (3000, g), (99999, g)]
    bad = [(s, budget_weight(s, g, w, r), want) for s, want in pts
           if abs(budget_weight(s, g, w, r) - want) > 1e-9]
    check("T1  budget ramp: zero during warmup, linear, then constant",
          not bad, str(bad))
    check("T1b gamma=0 stays zero at every step",
          all(budget_weight(s, 0.0, w, r) == 0.0 for s in (1, 5000, 99999)))

    # ---- allocation statistics ------------------------------------------
    # vocab: ids 0..119 are group elements, 120+ are specials.
    lut = torch.zeros(127, dtype=torch.long)
    lut[0] = 0     # identity
    lut[1] = 1     # a transposition
    lut[2] = 4     # a 5-cycle
    lut[3] = 2

    ids = torch.tensor([[1, 2, 120, 1]])          # 120 is a special token
    # perfect allocation on the three group tokens, junk on the special
    n_t = torch.tensor([[1.0, 4.0, 3.0, 1.0]])
    st = allocation_stats(n_t, ids, lut, max_order=4)
    check("T2  special tokens excluded from the statistics",
          st["n_special_excluded"] == 1 and st["n_eval"] == 3,
          f"excluded={st['n_special_excluded']} eval={st['n_eval']}")
    check("T2b mean cost ignores the special token",
          abs(st["mean_n_t"] - 2.0) < 1e-6, f"{st['mean_n_t']}")
    check("T3  perfect allocation: mae 0, no under, no over",
          st["mae"] == 0.0 and st["under_frac"] == 0.0 and st["over_frac"] == 0.0,
          f"mae={st['mae']} under={st['under_frac']}")

    # demand must be CLIPPED to what the model can express
    st2 = allocation_stats(torch.tensor([[2.0]]), torch.tensor([[2]]), lut,
                           max_order=2)
    check("T4  demand clipped to max_order (5-cycle at K=2 demands 2, not 4)",
          st2["mae"] == 0.0, f"mae={st2['mae']}")

    # under vs over must not be collapsed into one magnitude
    ids3 = torch.tensor([[1, 2, 3]])
    under = allocation_stats(torch.tensor([[0.0, 3.0, 1.0]]), ids3, lut, 4)
    over = allocation_stats(torch.tensor([[2.0, 4.0, 3.0]]), ids3, lut, 4)
    check("T5  under-allocation detected separately from over-allocation",
          under["under_frac"] == 1.0 and under["over_frac"] == 0.0
          and over["under_frac"] == 0.0 and abs(over["over_frac"] - 2 / 3) < 1e-6,
          f"under={under['under_frac']}/{under['over_frac']} "
          f"over={over['under_frac']}/{over['over_frac']}")

    # per-demand split: a gate that closes UNIFORMLY must not look like success
    flat = allocation_stats(torch.tensor([[2.0, 2.0, 2.0]]), ids3, lut, 4)
    means = {d: v["mean_n_t"] for d, v in flat["by_demand"].items()}
    check("T6  uniform allocation shows identical means across demands",
          set(means.values()) == {2.0} and len(means) == 3, str(means))
    check("T6b uniform allocation has no correlation with demand",
          flat["corr"] is None, f"corr={flat['corr']}")

    # a gate that tracks demand must show it
    tracking = allocation_stats(torch.tensor([[1.0, 4.0, 2.0]]), ids3, lut, 4)
    check("T7  demand-tracking allocation correlates at 1.0",
          tracking["corr"] is not None and abs(tracking["corr"] - 1.0) < 1e-6,
          f"corr={tracking['corr']}")

    # ---- verdict logic ---------------------------------------------------
    orc = dict(token_acc=0.9999)
    mk = lambda acc, cost: dict(token_acc=acc, alloc=dict(mean_n_t=cost))
    check("T8  collapsed accuracy reported as budget-too-strong",
          "BUDGET TOO STRONG" in gate_verdict(mk(0.05, 0.1), orc, 4))
    check("T8b cost still at K reported as gate-did-not-close",
          "DID NOT CLOSE" in gate_verdict(mk(0.9999, 3.95), orc, 4))
    check("T8c oracle accuracy at reduced cost reported as useful",
          "USEFUL ALLOCATION" in gate_verdict(mk(0.9999, 1.35), orc, 4))

    # ---- escape fraction, not a mean and not the best run ----------------
    mkr = lambda seed, cost, acc=0.999: dict(
        seed=seed, n_h=4, token_acc=acc, alloc=dict(mean_n_t=cost))
    split = [mkr(1, 1.298, 0.9999), mkr(2, 4.0, 0.9995), mkr(3, 4.0, 0.9997)]
    check("T12 one-of-three split reported as 1 of 3, not as its mean",
          closure_fraction(split) == (1, 3), str(closure_fraction(split)))
    check("T12b the best-ACCURACY run is not what decides closure",
          not is_closed(max(split, key=lambda r: r["token_acc"]))
          or is_closed(split[0]),
          "closure must follow cost, not accuracy")
    check("T12c a run still at full order does not count as closed",
          not is_closed(mkr(9, 4.0)) and not is_closed(mkr(9, 3.9)))
    check("T12d a run at oracle cost counts as closed",
          is_closed(mkr(9, 1.298)))
    check("T12e all-closed and none-closed report cleanly",
          closure_fraction([mkr(1, 1.3), mkr(2, 1.3)]) == (2, 2)
          and closure_fraction([mkr(1, 4.0)]) == (0, 1))

    # ---- optimizer split and the gate's input-dependence ------------------
    # A stub standing in for the fla layer: the real one is a CUDA module, but
    # everything under test here is name/identity plumbing plus gating.py,
    # which is CPU-only.
    import torch.nn as nn
    from gating import HaltingGate

    class _StubLayer(nn.Module):
        def __init__(self, hidden=16, K=4, bias=4.0):
            super().__init__()
            self.adaptive_order = True
            self.other = nn.Linear(hidden, hidden)
            self.halting_gate = HaltingGate(hidden_size=hidden, max_order=K,
                                            init_open_bias=bias)

    class _StubModel(nn.Module):
        def __init__(self, **kw):
            super().__init__()
            self.embed = nn.Embedding(32, 16)
            self.layers = nn.ModuleList([_StubLayer(**kw)])

    stub = _StubModel()
    names, gp = gate_parameters(stub)
    n_gate_expected = len(list(stub.layers[0].halting_gate.parameters()))
    check("T9  gate parameters found and named",
          len(gp) == n_gate_expected and all("halting_gate" in n for n in names),
          f"{len(gp)} found (expected {n_gate_expected}): {names}")

    groups = param_groups(stub, lr=1e-3, gate_lr=1e-2)
    total = sum(len(g["params"]) for g in groups)
    all_ids = [id(p) for g in groups for p in g["params"]]
    check("T9b split covers every parameter exactly once",
          total == len(list(stub.parameters())) and len(set(all_ids)) == total,
          f"{total} in groups vs {len(list(stub.parameters()))} in model")
    gate_group = [g for g in groups if g["lr"] == 1e-2]
    check("T9c the gate group carries the gate learning rate",
          len(gate_group) == 1 and len(gate_group[0]["params"]) == n_gate_expected)
    check("T9d gate_lr=None gives a single group at the base lr",
          len(param_groups(stub, 1e-3, None)) == 1)

    # the dangerous silent failure: a gate-less model must REFUSE --gate_lr
    # rather than quietly training everything at the base rate.
    plain = nn.Sequential(nn.Linear(4, 4))
    try:
        param_groups(plain, 1e-3, 1e-2)
        raised = False
    except RuntimeError:
        raised = True
    check("T9e --gate_lr on a model with no gate raises instead of no-op", raised)

    # ---- the saturation diagnostic ---------------------------------------
    w0 = gate_weight_norm(stub)
    check("T10 gate output weight starts at exactly zero (input-independent)",
          w0 == 0.0, f"|W| = {w0}")

    x = torch.randn(2, 6, 16)
    g_a, n_a = stub.layers[0].halting_gate(x)
    g_b, n_b = stub.layers[0].halting_gate(torch.randn(2, 6, 16))
    check("T10b at init every token gets the SAME order regardless of input",
          abs(n_a.std().item()) < 1e-6 and abs(n_a.mean().item() - n_b.mean().item()) < 1e-6,
          f"std {n_a.std().item():.2e}, means {n_a.mean().item():.4f} vs "
          f"{n_b.mean().item():.4f}")

    with torch.no_grad():
        nn.init.normal_(stub.layers[0].halting_gate.mlp[-1].weight, std=1.0)
    check("T10c |W| becomes non-zero once the weight moves",
          gate_weight_norm(stub) > 0.0)
    check("T10d and the order then varies across tokens",
          stub.layers[0].halting_gate(x)[1].std().item() > 1e-3)

    # init_open_bias must actually reach the gate, or the flag measures nothing
    lo = _StubModel(bias=0.0).layers[0].halting_gate
    hi = _StubModel(bias=4.0).layers[0].halting_gate
    n_lo = lo(x)[1].mean().item()
    n_hi = hi(x)[1].mean().item()
    check("T11 lower init bias gives a lower starting order",
          n_lo < n_hi - 0.5, f"bias 0.0 -> n_t {n_lo:.3f}, bias 4.0 -> {n_hi:.3f}")

    # ---- start-order guard ------------------------------------------------
    w_rep, _ = expected_start_order("repaired", 4.0, 4)
    w_hf, t_hf = expected_start_order("hf_default", 4.0, 4)
    check("T14 repaired gate at bias 4 starts at 3.8233",
          abs(w_rep - 3.8233) < 1e-4, f"{w_rep:.4f}")
    check("T14b hf_default ignores the bias and starts at 0.9375",
          abs(w_hf - 0.9375) < 1e-9
          and expected_start_order("hf_default", 2.0, 4)[0] == w_hf)
    # the logged starts of gate_bias40_more must sit inside the hf band and
    # the open-gate starts must not: the guard has to tell the two apart
    check("T14c guard accepts every logged pre-repair start (0.9333-0.9445)",
          all(abs(v - w_hf) <= t_hf for v in (0.9333, 0.9361, 0.9445, 0.9392,
                                              0.9348, 0.9398, 0.9357)))
    check("T14d guard rejects an open gate when hf_default was asked for",
          abs(3.8233 - w_hf) > t_hf and abs(3.8125 - w_hf) > t_hf)
    check("T14e guard rejects a clobbered gate when repaired was asked for",
          abs(0.9375 - w_rep) > expected_start_order("repaired", 4.0, 4)[1])
    try:
        expected_start_order("legacy", 4.0, 4)
        raised = False
    except ValueError:
        raised = True
    check("T14f an unknown --gate_init raises instead of guessing", raised)
    check("T14g hard gates from step 1, open start: order exactly 4, not 3.8233",
          expected_start_order("repaired", 4.0, 4, hard=True)[0] == 4.0)
    check("T14h hard start at bias 0.5: sigmoid 0.62, products .62/.39 -> order 1",
          expected_start_order("repaired", 0.5, 4, hard=True)[0] == 1.0)
    try:
        expected_start_order("hf_default", 4.0, 4, hard=True)
        raised = False
    except ValueError:
        raised = True
    check("T14i pre-repair start with hard gates from step 1 is refused", raised)
    w_m1 = expected_start_order("repaired", 4.0, 4, min_order=1)[0]
    check("T14j min_order 1, open start, soft: 1 + .982 + .964 + .947 = 3.8933",
          abs(w_m1 - 3.8933) < 1e-4, f"{w_m1:.4f}")
    check("T14k min_order 1, pre-repair start, soft: 1 + .5 + .25 + .125 = 1.875",
          abs(expected_start_order("hf_default", 0.0, 4, min_order=1)[0] - 1.875) < 1e-9)
    lo_h, tol_h = expected_start_order("hf_default", 0.0, 4, hard=True, min_order=1)
    check("T14l min_order 1 makes the pre-repair hard start legal, in [1, 2]",
          abs(lo_h - 1.5) < 1e-9 and tol_h == 0.5)
    check("T14m min_order 1, open start, hard: still exactly 4",
          expected_start_order("repaired", 4.0, 4, hard=True, min_order=1)[0] == 4.0)

    # ---- fingerprint ------------------------------------------------------
    # real step-1 rows: gate_train_p10 seed 1 (n_t not logged) and
    # gate_bias40_more seed 4 (n_t and |W| logged)
    ref = [dict(mode="learned", seed=1, loss_trace=[[1, 4.8671, 0.0]]),
           dict(mode="oracle", seed=1, loss_trace=[[1, 4.8661, 0.0]]),
           dict(mode="learned", seed=4, loss_trace=[[1, 4.8723, 0.9333, 7.019031]])]
    fp = lambda *a, **k: fingerprint_check(ref, *a, **k)["verdict"]
    check("T15 p10 seed 1 reproduced exactly -> MATCH",
          fp("learned", 1, 4.8671, n_t=0.9375) == "MATCH")
    check("T15b the open-gate model's step-1 loss (4.8658) -> MISMATCH",
          fp("learned", 1, 4.8658) == "MISMATCH")
    check("T15c seed 2's smaller gap (4.8721 vs 4.8718) is still caught",
          fingerprint_check([dict(mode="learned", seed=2,
                                  loss_trace=[[1, 4.8718, 0.0]])],
                            "learned", 2, 4.8721)["verdict"] == "MISMATCH")
    check("T15d bias40 seed 4 with matching n_t and |W| -> MATCH",
          fp("learned", 4, 4.8723, n_t=0.9335, w=7.05) == "MATCH")
    check("T15e right loss but an open gate's n_t and |W| -> MISMATCH",
          fp("learned", 4, 4.8723, n_t=3.8233, w=0.248) == "MISMATCH")
    check("T15h n_t alone wrong (|W| right) -> MISMATCH",
          fp("learned", 4, 4.8723, n_t=3.8233, w=7.05) == "MISMATCH")
    check("T15i |W| alone wrong (n_t right) -> MISMATCH",
          fp("learned", 4, 4.8723, n_t=0.9335, w=0.248) == "MISMATCH")
    check("T15f oracle arm compared on loss only",
          fp("oracle", 1, 4.8661, n_t=9.9, w=99.0) == "MATCH")
    check("T15g a seed absent from the reference says so",
          fp("learned", 9, 4.87) == "NO REFERENCE")

    # ---- the logit wall -----------------------------------------------------
    z = torch.tensor([[-7.0, -5.0, 0.0, 4.9, 5.0, 6.0, 8.0]], requires_grad=True)
    wl = logit_wall_loss([z], 5.0)
    want = (2.0 ** 2 + 1.0 ** 2 + 3.0 ** 2) / 7       # only |z| > 5 counts
    wl.backward()
    want_g = torch.tensor([[-4.0, 0, 0, 0, 0, 2.0, 6.0]]) / 7
    check("T16 wall loss: zero inside [-c, c], squared excess outside",
          abs(wl.item() - want) < 1e-6, f"{wl.item():.6f} vs {want:.6f}")
    check("T16b wall gradient is 2(|z|-c)sign(z)/N outside, exactly 0 inside",
          torch.allclose(z.grad, want_g), f"{z.grad.tolist()}")
    zb = torch.tensor([10.0], dtype=torch.bfloat16, requires_grad=True)
    torch.sigmoid(zb).sum().backward()
    sig_g = zb.grad.item()
    zb2 = torch.tensor([10.0], dtype=torch.bfloat16, requires_grad=True)
    logit_wall_loss([zb2], 5.0).backward()
    check("T16c at z=10 in bf16 the sigmoid's gradient is 0 and the wall's is not",
          sig_g == 0.0 and zb2.grad.float().item() > 1.0,
          f"sigmoid grad {sig_g}, wall grad {zb2.grad.float().item()}")

    # the documented failure, reproduced, and the wall undoing it: a bf16 gate
    # with every logit near 10, trained on the budget alone
    def _sat_gate_run(wall, budget, steps=300):
        torch.manual_seed(0)
        gq = HaltingGate(hidden_size=16, max_order=4, init_open_bias=10.0,
                         fp32_halting=False).to(torch.bfloat16)
        with torch.no_grad():
            nn.init.normal_(gq.mlp[-1].weight, std=0.02)
        opt_q = torch.optim.AdamW(gq.parameters(), lr=1e-2)
        xq = torch.randn(8, 32, 16, dtype=torch.bfloat16)
        sat0 = None
        for _ in range(steps):
            _, nq = gq(xq)
            if sat0 is None:
                sat0 = (gq.last_lambdas == 1).float().mean().item()
            lq = torch.zeros(())
            if budget:
                lq = lq + nq.float().mean()
            if wall:
                lq = lq + logit_wall_loss([gq.last_logits], wall)
            opt_q.zero_grad(set_to_none=True)
            lq.backward()
            opt_q.step()
        with torch.no_grad():
            _, nq = gq(xq)
        return (nq.float().mean().item(), gq.last_logits.float().max().item(),
                sat0)
    n_off, z_off, s_off = _sat_gate_run(0.0, True)
    n_on, z_on, _ = _sat_gate_run(5.0, True)
    n_wall, z_wall, _ = _sat_gate_run(5.0, False)
    check("T17 setup: every halting probability starts at exactly 1.0 (bf16)",
          s_off == 1.0, f"saturated fraction {s_off}")
    check("T17b without the wall the budget cannot move a saturated gate",
          n_off == 4.0, f"n_t {n_off:.4f}, zmax {z_off:.2f}")
    check("T17c with the wall the same budget closes it",
          n_on < 1.0 and z_on <= 5.0 + 0.5, f"n_t {n_on:.4f}, zmax {z_on:.2f}")
    check("T17d side effect, recorded: the wall alone lowers n_t (no task loss)",
          n_wall < 3.5 and z_wall <= 5.0 + 0.5, f"n_t {n_wall:.4f}, zmax {z_wall:.2f}")

    # the wall must refuse to be a silent no-op
    fresh_stub = _StubModel(bias=4.0)
    try:
        gate_logits(fresh_stub)
        raised = False
    except RuntimeError:
        raised = True
    check("T17e gate_logits raises before any forward instead of returning []",
          raised)
    nogate = nn.Sequential(nn.Linear(4, 4))
    try:
        gate_logits(nogate)
        raised = False
    except RuntimeError:
        raised = True
    check("T17f gate_logits raises on a model with no gate", raised)
    _ = fresh_stub.layers[0].halting_gate(torch.randn(2, 5, 16))
    zl = gate_logits(fresh_stub)
    check("T17g after a forward the logits come back with their graph",
          len(zl) == 1 and zl[0].requires_grad)
    zm, sf = saturation_stats(fresh_stub)
    check("T17h saturation stats on an open fp32 gate: zmax 4, nothing saturated",
          abs(zm - 4.0) < 1e-5 and sf == 0.0, f"zmax {zm}, sat {sf}")

    # ---- k/n with intervals ---------------------------------------------------
    lo3, hi3 = wilson(3, 10)
    lo0, hi0 = wilson(0, 20)
    check("T18 Wilson 3/10 = [0.108, 0.603] (PLAN quotes [0.11, 0.60])",
          abs(lo3 - 0.1078) < 1e-3 and abs(hi3 - 0.6032) < 1e-3,
          f"[{lo3:.4f}, {hi3:.4f}]")
    check("T18b Wilson 0/20 = [0, 0.161]: zero successes is not zero rate",
          lo0 == 0.0 and abs(hi0 - 0.1611) < 1e-3, f"[{lo0:.4f}, {hi0:.4f}]")
    mk = lambda acc, corr, d1, d4: dict(
        token_acc=acc, n_h=4,
        alloc=dict(corr=corr, mean_n_t=0.9 * d1 + 0.1 * d4,
                   by_demand={"1": dict(mean_n_t=d1), "4": dict(mean_n_t=d4)}))
    rows_t = [mk(0.9999, 1.0, 1.0, 4.0),    # exact
              mk(0.9999, 1.0, 2.0, 4.0),    # separated at 2.199, not exact
              mk(0.8937, 0.91, 1.2, 4.0),   # lost the task: neither
              mk(0.7944, 1.0, 1.0, 4.0),    # exact-looking but task lost: neither
              mk(0.9995, 0.0, 4.0, 4.0)]    # never closed
    s = arm_summary(rows_t)
    check("T18c summary: 2 separated, 1 exact, of 5 -- accuracy gates both",
          s["n"] == 5 and s["separated"] == 2 and s["exact"] == 1,
          f"{s['separated']} sep, {s['exact']} exact")

    # ---- demand model -------------------------------------------------------
    els = _s5_elements()
    ell_all = demand_from_alphabet([], "ell")
    check("T20 S5 census: identity first, 10 transpositions, 24 five-cycles",
          ell_all[0] == 0 and ell_all.count(1) == 10 and ell_all.count(4) == 24)
    five = [i for i, e in enumerate(ell_all) if e == 4]
    swaps = [i for i, e in enumerate(ell_all) if e == 1]
    g5 = _generated_group(five)
    check("T20b 5-cycles generate a group of order 60, all even (A5)",
          len(g5) == 60 and all(ell_all[i] % 2 == 0 for i in g5), f"order {len(g5)}")
    d_a5 = demand_from_alphabet(five, "model")
    check("T20c under an A5 alphabet every non-identity element costs 2, a 5-cycle too",
          d_a5[0] == 0 and all(d_a5[i] == 2 for i in range(1, 120))
          and all(d_a5[i] == 2 for i in five))
    check("T20d under an S5 alphabet (swaps + 5-cycles) the demand is ell",
          demand_from_alphabet(swaps + five, "model") == ell_all)
    check("T20e transpositions alone generate S5, so demand is ell",
          len(_generated_group(swaps)) == 120
          and demand_from_alphabet(swaps, "model") == ell_all)
    check("T20f rule 'ell' still charges a 5-cycle 4 under the A5 alphabet",
          all(demand_from_alphabet(five, "ell")[i] == 4 for i in five))
    three = [i for i, e in enumerate(ell_all) if e == 2
             and sum(els[i][j] != j for j in range(5)) == 3][:1]
    try:
        demand_from_alphabet(three, "model")
        raised = False
    except ValueError:
        raised = True
    check("T20g a single 3-cycle generates a group of order 3: refused, not guessed",
          raised and len(_generated_group(three)) == 3)
    check("T20h alphabet from metadata: p=1.0 -> 24 five-cycles; p=0.1 -> 34 tokens",
          len(alphabet_from_meta(dict(p_hard=1.0, easy_ell=1, hard_ell=4))) == 24
          and len(alphabet_from_meta(dict(p_hard=0.1, easy_ell=1, hard_ell=4))) == 34)

    # ---- hard gates in training, and the soft-gate evaluation ---------------
    check("T19 hard-gate schedule: never when hard_from = 0",
          not any(gates_hard_at(s_, 0) for s_ in (1, 4000, 12000, 20000)))
    check("T19b hard from exactly step 12000, not one step earlier",
          not gates_hard_at(11999, 12000) and gates_hard_at(12000, 12000)
          and gates_hard_at(20000, 12000))
    n_switch = sum(gates_hard_at(s_, 12000) != gates_hard_at(s_ - 1, 12000)
                   for s_ in range(1, 20001))
    check("T19c the switch happens once in a 20 000-step run, and never at 0",
          n_switch == 1 and sum(gates_hard_at(s_, 0) != gates_hard_at(s_ - 1, 0)
                                for s_ in range(1, 20001)) == 0)

    import types as _types
    import torch.nn.functional as _F

    class _FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.adaptive_order = True
            self.gate_hard = True          # left HARD, as training's end might
            self.last_n_t = None

    class _FakeLM(nn.Module):
        """Right only with soft gates: the failure being diagnosed, in miniature."""
        def __init__(self, V=5):
            super().__init__()
            self.layer = _FakeLayer()
            self.V = V

        def forward(self, input_ids):
            y_ = input_ids % self.V
            if self.layer.gate_hard:
                y_ = (y_ + 1) % self.V
            self.layer.last_n_t = torch.full(input_ids.shape,
                                             1.0 if self.layer.gate_hard else 4 / 3)
            return _types.SimpleNamespace(logits=_F.one_hot(y_, self.V).float())

    fake = _FakeLM()
    fx = torch.randint(0, 50, (300, 8))
    fy = fx % 5
    ev_soft = evaluate_gates(fake, fx, fy, hard=False)
    ev_hard = evaluate_gates(fake, fx, fy, hard=True)
    check("T19d soft evaluation sets soft mode itself: token 1.0, n_t 4/3",
          ev_soft["token_acc"] == 1.0 and ev_soft["seq_acc"] == 1.0
          and abs(ev_soft["mean_n_t_all"] - 4 / 3) < 1e-6, str(ev_soft))
    check("T19e hard evaluation of the same model: token 0.0, n_t 1.0",
          ev_hard["token_acc"] == 0.0 and abs(ev_hard["mean_n_t_all"] - 1.0) < 1e-6,
          str(ev_hard))
    try:
        evaluate_gates(nn.Linear(2, 2), fx, fy, hard=False)
        raised = False
    except RuntimeError:
        raised = True
    check("T19f evaluating a model with no adaptive layer raises", raised)

    rows_s = [dict(r, soft_eval=dict(token_acc=a))
              for r, a in zip(rows_t, (0.9999, 0.9999, 0.9990, 0.9995, 0.9999))]
    s2 = arm_summary(rows_s)
    check("T19g summary under soft evaluation: corr-1.0 runs with soft acc >= 0.99",
          s2["separated_soft"] == 3 and s2["separated"] == 2 and s2["has_soft"],
          f"{s2['separated_soft']} soft, {s2['separated']} hard")
    check("T19h runs without a soft evaluation never count as soft-separated",
          arm_summary(rows_t)["separated_soft"] == 0
          and not arm_summary(rows_t)["has_soft"])

    # ---- saving a trained model --------------------------------------------
    nm = checkpoint_name(dict(arm="learned(g=0.1,hard@1)", gamma=0.1, seed=11))
    check("T21 checkpoint name is file-system safe and keeps arm, gamma, seed",
          nm == "learned_g_0_1_hard_1_g0.1_seed11.pt", nm)
    nm2 = checkpoint_name(dict(arm="learned(g=0.1,hard@1)", gamma=0.3, seed=11))
    check("T21b two penalty weights never share a file", nm2 != nm, nm2)

    print("\n  " + ("all self-tests passed" if ok else "SELF-TEST FAILED"))
    return ok


# ===========================================================================
# Training
# ===========================================================================

def run_arm(name, n_h, mode, gamma, data, lut, args, seed, wall=0.0,
            hard_from=0, dlut=None):
    """mode: 'fixed' | 'oracle' | 'learned'."""
    from exp_matched_compute import build
    from oracle_order import oracle_gates, set_external_gates

    tr_x, tr_y, te_x, te_y = data
    adaptive = mode in ("oracle", "learned")
    model = build(n_h, adaptive, args.n_heads, args.head_dim, args.hidden,
                  args.n_layers, args.vocab, seed,
                  gate_init_bias=args.gate_init_bias,
                  gate_init=args.gate_init, min_order=args.min_order)
    if mode == "learned":
        set_gate_hard(model, False)          # soft while training
    # Only the learned arm has a gate to give its own learning rate. The oracle
    # arm has a gate object but never uses it -- its gates are supplied
    # externally -- so raising its learning rate would change nothing while
    # making the two arms look differently configured.
    gate_lr = args.gate_lr if mode == "learned" else None
    opt = torch.optim.AdamW(param_groups(model, args.lr, gate_lr))
    g = torch.Generator(device=DEV).manual_seed(seed)

    trace = []
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        if mode == "learned" and (gates_hard_at(step, hard_from)
                                  != gates_hard_at(step - 1, hard_from)):
            set_gate_hard(model, True)
            print(f"      step {step:>6}  gates now HARD (straight-through) for "
                  f"the rest of training")
        idx = torch.randint(0, tr_x.shape[0], (args.batch,), device=DEV, generator=g)
        x, y = tr_x[idx], tr_y[idx]
        if mode == "oracle":
            set_external_gates(model, oracle_gates(x, lut, n_h, dtype=DTYPE))

        logits = model(input_ids=x).logits
        task = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
        loss, gw, bud = task, 0.0, 0.0
        if mode == "learned":
            gw = budget_weight(step, gamma, args.budget_warmup, args.budget_ramp)
            # measured every step, not only while the penalty is active: the
            # warmup phase is where the gate does or does not become
            # input-dependent, and a log that reads 0.000 there hides it.
            mean_orders = torch.stack([n.float().mean() for n in collect_n_t(model)])
            bud_t = mean_orders.mean()
            if gw > 0.0:
                loss = task + gw * bud_t
            if wall > 0.0:
                # from step 1, not ramped: the saturation it prevents happens
                # inside the penalty-free warmup
                loss = loss + args.logit_wall_weight * logit_wall_loss(
                    gate_logits(model), wall)
            bud = float(bud_t.detach().item())
            if step == 1:
                want, tol = expected_start_order(
                    args.gate_init, args.gate_init_bias, n_h,
                    hard=gates_hard_at(1, hard_from), min_order=args.min_order)
                if abs(bud - want) > tol:
                    raise SystemExit(
                        f"gate did not start where --gate_init {args.gate_init} "
                        f"says it should: mean n_t {bud:.4f} at step 1, expected "
                        f"{want:.4f} +/- {tol:g}. The model being trained is not "
                        f"the one the log header describes.")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        dense = step <= args.log_dense_until and step % 250 == 0
        if step % args.log_every == 0 or step == 1 or dense:
            extra = ""
            wn = None
            zmax = sat = None
            if mode == "learned":
                wn = gate_weight_norm(model)
                zmax, sat = saturation_stats(model)
                extra = (f"  gamma_eff {gw:.4f}  mean n_t {bud:.3f}"
                         f"  gate |W| {wn:.3e}  zmax {zmax:.2f}  sat {sat:.3f}")
            trace.append((step, round(task.item(), 4), round(bud, 4),
                          None if wn is None else round(wn, 6),
                          None if zmax is None else round(zmax, 4),
                          None if sat is None else round(sat, 4)))
            print(f"      step {step:>6}  task {task.item():.4f}{extra}")
            if step == 1 and args.ref_rows is not None:
                fp = fingerprint_check(args.ref_rows, mode, seed,
                                       round(task.item(), 4),
                                       round(bud, 4) if mode == "learned" else None,
                                       wn)
                print(f"      fingerprint vs {Path(args.fingerprint).name}: "
                      f"{fp['verdict']}   {fp['detail']}")
    dt = time.perf_counter() - t0
    if mode == "oracle":
        set_external_gates(model, None)
    fingerprint = fp if (args.ref_rows is not None) else None

    # ---- evaluation: hard gates, so the cost is an integer factor count ----
    if mode == "learned":
        set_gate_hard(model, True)
    model.eval()
    tc = tt = sc = ns = 0
    alloc_acc = None
    with torch.no_grad():
        for i in range(0, te_x.shape[0], 128):
            x, y = te_x[i:i + 128], te_y[i:i + 128]
            if mode == "oracle":
                set_external_gates(model, oracle_gates(x, lut, n_h, dtype=DTYPE))
            pred = model(input_ids=x).logits.float().argmax(-1)
            eq = pred == y
            tc += eq.sum().item(); tt += eq.numel()
            sc += eq.all(-1).sum().item(); ns += y.shape[0]
            if adaptive:
                n_t = collect_n_t(model)[0].detach()
                st = allocation_stats(n_t, x, lut if dlut is None else dlut, n_h)
                if alloc_acc is None:
                    alloc_acc = {k: [v] for k, v in st.items()}
                else:
                    for k, v in st.items():
                        alloc_acc[k].append(v)
    if mode == "oracle":
        set_external_gates(model, None)

    # merge per-batch allocation stats
    if alloc_acc is None:
        alloc = dict(mean_n_t=float(n_h), by_demand={}, mae=None,
                     under_frac=None, over_frac=None, corr=None,
                     n_eval=0, n_special_excluded=0)
    else:
        nev = sum(alloc_acc["n_eval"])
        wmean = lambda key: (
            sum(v * n for v, n in zip(alloc_acc[key], alloc_acc["n_eval"])
                if v is not None) / nev) if nev else None
        by = {}
        for d in sorted({d for b in alloc_acc["by_demand"] for d in b}):
            tot = sum(b[d]["n"] for b in alloc_acc["by_demand"] if d in b)
            s = sum(b[d]["mean_n_t"] * b[d]["n"] for b in alloc_acc["by_demand"] if d in b)
            by[d] = dict(n=tot, mean_n_t=s / tot if tot else None)
        alloc = dict(n_eval=nev,
                     n_special_excluded=sum(alloc_acc["n_special_excluded"]),
                     mean_n_t=wmean("mean_n_t"), mean_demand=wmean("mean_demand"),
                     by_demand=by, mae=wmean("mae"),
                     under_frac=wmean("under_frac"), over_frac=wmean("over_frac"),
                     corr=wmean("corr"))

    # The hard evaluation above is the reported number. The soft one shows how
    # much of any accuracy loss comes from switching soft gates to {0, 1} at
    # evaluation rather than from the trained model.
    soft_eval = None
    if mode == "learned":
        soft_eval = evaluate_gates(model, te_x, te_y, hard=False)
        print(f"      eval   hard gates: token {tc / tt:.4f}   |   soft gates: "
              f"token {soft_eval['token_acc']:.4f}, seq {soft_eval['seq_acc']:.4f}, "
              f"mean n_t {soft_eval['mean_n_t_all']:.3f}")

    row = dict(arm=name, mode=mode, n_h=n_h, gamma=gamma, seed=seed,
               token_acc=tc / tt, seq_acc=sc / max(1, ns), train_s=dt,
               gate_lr=gate_lr, gate_init=args.gate_init,
               gate_init_bias=(None if args.gate_init == "hf_default"
                               else args.gate_init_bias),
               fingerprint=fingerprint,
               logit_wall=(wall if mode == "learned" else None),
               min_order=(args.min_order if mode == "learned" else None),
               hard_from=(hard_from if mode == "learned" else None),
               demand_rule=args.demand,
               soft_eval=soft_eval,
               logit_wall_weight=(args.logit_wall_weight
                                  if mode == "learned" and wall > 0 else None),
               gate_w_final=(gate_weight_norm(model) if mode == "learned" else None),
               loss_trace=trace, alloc=alloc)
    if getattr(args, "save_dir", None) and mode == "learned":
        row["checkpoint"] = save_checkpoint(model, row, args, n_h, adaptive, seed)
        print(f"      saved  {row['checkpoint']}")
    del model
    torch.cuda.empty_cache()
    return row


def checkpoint_name(row) -> str:
    """File-system-safe name for a run: arm, penalty weight, seed."""
    import re
    arm = re.sub(r"[^A-Za-z0-9]+", "_", str(row["arm"])).strip("_")
    return f"{arm}_g{row['gamma']:g}_seed{row['seed']}.pt"


def save_checkpoint(model, row, args, n_h, adaptive, seed) -> str:
    """Everything needed to rebuild the model exactly and to check the load:
    the weights, the arguments `build` was called with, where the data came
    from, and the accuracy and cost this run was evaluated at."""
    sd = Path(args.save_dir)
    sd.mkdir(parents=True, exist_ok=True)
    path = sd / checkpoint_name(row)
    torch.save(dict(
        state_dict=model.state_dict(),
        build=dict(n_h=n_h, adaptive=adaptive, n_heads=args.n_heads,
                   head_dim=args.head_dim, hidden=args.hidden,
                   n_layers=args.n_layers, vocab=args.vocab, seed=seed,
                   gate_init_bias=args.gate_init_bias, gate_init=args.gate_init,
                   min_order=args.min_order),
        data=dict(p=args.p, k=args.k, data_dir=args.data_dir,
                  n_train=args.n_train, n_test=args.n_test),
        result=dict(token_acc=row["token_acc"], seq_acc=row["seq_acc"],
                    mean_n_t=row["alloc"]["mean_n_t"], gamma=row["gamma"]),
    ), path)
    return str(path)


def report(rows, args):
    print("=" * 78)
    print(f"{'arm':<30} {'seed':>4} {'token':>7} {'soft':>7} {'seq':>6} {'cost':>7} "
          f"{'mae':>5} {'under':>6} {'corr':>5}")
    print("-" * 78)
    for r in rows:
        a = r["alloc"]
        f = lambda v, p=3: ("  n/a" if v is None else f"{v:.{p}f}")
        se = r.get("soft_eval")
        soft_txt = "    n/a" if not se else f"{se['token_acc']:>7.4f}"
        print(f"{r['arm']:<30} {r['seed']:>4} {r['token_acc']:>7.4f} {soft_txt} "
              f"{r['seq_acc']:>6.3f} {f(a['mean_n_t']):>7} {f(a['mae'],2):>5} "
              f"{f(a['under_frac'],2):>6} {f(a['corr'],2):>5}")
    print("=" * 78)

    # Per-arm fractions first: they need no oracle, and they are the numbers
    # that go in a document. Seeds, not repeats, are the unit here.
    keys = sorted({(r["gamma"], r.get("logit_wall") or 0.0, r.get("hard_from") or 0)
                   for r in rows if r["mode"] == "learned"})
    in_arm = lambda r, g_, w_, h_: (r["mode"] == "learned" and r["gamma"] == g_
                                    and (r.get("logit_wall") or 0.0) == w_
                                    and (r.get("hard_from") or 0) == h_)
    if keys:
        print("  LEARNED-GATE ARMS  (separated = corr >= 0.99 and token >= 0.99;")
        print("                      exact = every class gets min(D, K), token >= 0.99)")
        for g_, w_, h_ in keys:
            arm = [r for r in rows if in_arm(r, g_, w_, h_)]
            s = arm_summary(arm)
            print(f"  gamma {g_:g}  wall {w_:g}  hard from {h_ or 'never'}   "
                  f"separated {s['separated']}/{s['n']} "
                  f"[{s['sep_ci'][0]:.2f}, {s['sep_ci'][1]:.2f}]   exact "
                  f"{s['exact']}/{s['n']} [{s['exact_ci'][0]:.2f}, {s['exact_ci'][1]:.2f}]")
            if s["has_soft"]:
                print(f"      separated if judged by SOFT-gate accuracy: "
                      f"{s['separated_soft']}/{s['n']} "
                      f"[{s['sep_soft_ci'][0]:.2f}, {s['sep_soft_ci'][1]:.2f}]")
            print("      costs, sorted: " + " ".join(f"{c:.4f}" for c in s["costs"]))
        print()

    orc = [r for r in rows if r["mode"] == "oracle"]
    if not orc:
        print("  NO VERDICT: no oracle arm in this run to compare against.")
        return
    best_orc = max(orc, key=lambda r: r["token_acc"])
    print(f"  oracle reference: token {best_orc['token_acc']:.4f} at cost "
          f"{best_orc['alloc']['mean_n_t']:.3f}\n")

    for gamma, w_, h_ in keys:
        arms = sorted([r for r in rows if in_arm(r, gamma, w_, h_)],
                      key=lambda r: r["seed"])
        n_closed, n_run = closure_fraction(arms)
        print(f"  gamma = {gamma:g}  wall = {w_:g}  hard from {h_ or 'never'}   "
              f"gate closed on {n_closed} of {n_run} run(s)")
        print("    per-seed cost — " + " · ".join(
            f"seed {r['seed']}: "
            + ("n/a" if r["alloc"]["mean_n_t"] is None
               else f"{r['alloc']['mean_n_t']:.3f}") for r in arms))
        if 0 < n_closed < n_run:
            print("    OUTCOME IS SPLIT across seeds. Report this as a fraction;")
            print("    a mean over these runs describes no run that happened.")
        for r in arms:
            print(f"    seed {r['seed']}: {gate_verdict(r, best_orc, r['n_h'])}")
            by = r["alloc"]["by_demand"]
            if not by:
                continue
            spread = " · ".join(f"D={d}: n_t {v['mean_n_t']:.2f}" for d, v in by.items())
            print(f"      allocation by true demand — {spread}")
            vals = [v["mean_n_t"] for v in by.values()]
            if (len(vals) > 1 and max(vals) - min(vals) < 0.05
                    and is_closed(r)):
                print("      !! allocation is FLAT across demands: this cost was")
                print("         reached by closing uniformly, not by tracking demand.")
        print()

    print("  Cost is required work per token. Path A computes all K factors and")
    print("  multiplies the disabled ones by zero, so none of this is wall-clock.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, default=0.1)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--data_dir", default="state_tracking/data")
    ap.add_argument("--n_train", type=int, default=100000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_heads", type=int, default=12)
    ap.add_argument("--head_dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=1)
    ap.add_argument("--max_order", type=int, default=4)
    ap.add_argument("--gammas", default="0,0.01,0.03,0.1",
                    help="budget weights to sweep; 0 is the gate-present control")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--budget_warmup", type=int, default=4000,
                    help="steps with no budget penalty (gate starts open)")
    ap.add_argument("--budget_ramp", type=int, default=4000,
                    help="steps to ramp the penalty linearly to gamma")
    ap.add_argument("--gate_lr", type=float, default=None,
                    help="learning rate for the halting gate alone; default "
                         "None means the gate trains at --lr like everything "
                         "else, which is how every run before 15 Sept was done")
    ap.add_argument("--gate_init_bias", type=float, default=4.0,
                    help="halting-gate output bias at init. 4.0 (the previous "
                         "default) gives sigmoid 0.982, derivative 0.018; 2.0 "
                         "gives 0.881 and derivative 0.105")
    ap.add_argument("--gate_init", choices=("repaired", "hf_default"),
                    default="repaired",
                    help="'repaired' (default): zero output weight, bias "
                         "--gate_init_bias, float32 gate -- every run from "
                         "gate_fixedinit on. 'hf_default': the pre-repair start "
                         "(bias 0, small random weights, bf16), to re-run "
                         "gate_train_p10 / gate_bias40_more")
    ap.add_argument("--logit_walls", default="0",
                    help="comma list of logit-wall positions c; each learned "
                         "arm is run once per value. 0 = no wall")
    ap.add_argument("--demand", choices=("ell", "model"), default="ell",
                    help="how allocation is scored: 'ell' = transposition length "
                         "(every run before 8 Oct); 'model' = the demand model, "
                         "which charges 2 for every element when the alphabet "
                         "generates A5. The oracle always uses 'ell'")
    ap.add_argument("--min_order", type=int, default=0,
                    help="first N factors always on; the gate decides only the "
                         "rest. 1 = never below plain DeltaNet. 0 = off, as in "
                         "every run before 6 Oct")
    ap.add_argument("--hard_froms", default="0",
                    help="comma list of steps from which training uses hard "
                         "(straight-through) gates; each learned arm runs once "
                         "per value. 0 = soft gates throughout")
    ap.add_argument("--logit_wall_weight", type=float, default=1.0,
                    help="rho in rho * mean(relu(|z| - c)^2)")
    ap.add_argument("--fingerprint", default=None,
                    help="results json of the run being reproduced; step-1 "
                         "values are compared per seed and MATCH/MISMATCH logged")
    ap.add_argument("--arms", default="fixed,oracle,learned")
    ap.add_argument("--log_every", type=int, default=2000)
    ap.add_argument("--log_dense_until", type=int, default=0,
                    help="also log every 250 steps up to this step, to see "
                         "what the gate does during the penalty-free warmup")
    ap.add_argument("--out", default="results/gate_train.json")
    ap.add_argument("--save_dir", default=None,
                    help="save each learned arm's trained model here (weights, "
                         "build settings, data settings and its evaluated "
                         "accuracy and cost), for work/check_ragged_trained.py")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        print("=" * 78)
        print("SELF-TEST — allocation statistics and schedule, known answers")
        print("=" * 78)
        raise SystemExit(0 if selftest() else 1)

    args.vocab = N_GROUP + N_SPECIAL
    args.ref_rows = None
    if args.fingerprint:
        args.ref_rows = json.loads(Path(args.fingerprint).read_text())["results"]
    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA")

    from exp_matched_compute import load
    from oracle_order import load_ell_lookup

    gammas = [float(x) for x in args.gammas.split(",")]
    walls = [float(x) for x in args.logit_walls.split(",")]
    hard_froms = [int(x) for x in args.hard_froms.split(",")]
    for h_ in hard_froms:
        if h_ < 0 or h_ > args.steps:
            raise SystemExit(f"--hard_froms {h_}: must be 0 (never) or a step "
                             f"in 1..{args.steps}")
    if args.gate_init == "hf_default" and 1 in hard_froms and args.min_order == 0:
        raise SystemExit("--hard_froms 1 with --gate_init hf_default: every halting "
                         "sigmoid starts at ~0.5, so hard gates start the model at "
                         "order ~0. Use --gate_init repaired.")
    sat_limit = BF16_SAT_LOGIT if args.gate_init == "hf_default" else FP32_SAT_LOGIT
    for w_ in walls:
        if w_ < 0 or (w_ > 0 and w_ >= sat_limit - 0.5):
            raise SystemExit(
                f"--logit_walls {w_:g}: a wall must sit clearly below the logit at "
                f"which this gate's sigmoid saturates ({sat_limit}), or it cannot "
                f"prevent saturation. Use e.g. 5.")
    seeds = [int(x) for x in args.seeds.split(",")]
    want = [a.strip() for a in args.arms.split(",")]

    group = f"S5_limit_to_het_p{int(round(args.p * 1000)):03d}"
    csv = Path(args.data_dir) / f"{group}={args.k}.csv"
    meta = Path(args.data_dir) / f"{group}_meta.json"
    for f in (csv, meta):
        if not f.exists():
            raise SystemExit(
                f"missing {f}\nGenerate it:\n"
                f"  python work/gen_heterogeneous.py --p {args.p} --k {args.k} "
                f"--samples {args.n_train + args.n_test}")
    m = json.loads(meta.read_text())
    lut = load_ell_lookup(meta, args.vocab, device=DEV)
    # the oracle keeps transposition length; the allocation statistics use the
    # demand rule chosen here
    alphabet = alphabet_from_meta(m)
    dvals = demand_from_alphabet(alphabet, args.demand)
    dlut = torch.zeros(args.vocab, dtype=torch.long)
    dlut[:120] = torch.tensor(dvals, dtype=torch.long)
    dlut = dlut.to(DEV)
    mean_demand = sum(dvals[i] for i in alphabet) / len(alphabet)
    data = load(csv, args.n_train, args.n_test)

    print("=" * 78)
    print("OBJECTIVE O4 — the halting gate, trained end-to-end")
    print("=" * 78)
    print(f"  data        {csv.name}   p = {args.p}")
    print(f"  demand      E[D] = {m['empirical_mean_ell']:.3f} by transposition length"
          f"   fixed order must provision {m['fixed_order_cost']}")
    print(f"  scoring     --demand {args.demand}: alphabet of {len(alphabet)} tokens "
          f"generates a group of order {len(_generated_group(alphabet))}; "
          f"per-token demand over the alphabet {sorted(set(dvals[i] for i in alphabet))}"
          f" (mean over alphabet tokens {mean_demand:.2f})")
    print(f"  budget      {args.steps} steps x batch {args.batch}, lr {args.lr}")
    print(f"  gammas      {', '.join(f'{g:g}' for g in gammas)}"
          f"   (0 = gate present, never penalised)")
    print(f"  schedule    penalty off for {args.budget_warmup} steps, then "
          f"ramped over {args.budget_ramp}")
    lr_txt = '= model lr' if args.gate_lr is None else f'{args.gate_lr:g}'
    if args.gate_init == "hf_default":
        print(f"  gate        init hf_default: bias 0 and N(0, 0.02) weights left by"
              f" post_init (n_t ~0.94 at init), bf16;")
        print(f"              --gate_init_bias IGNORED.   lr {lr_txt}")
    else:
        print(f"  gate        init repaired, bias {args.gate_init_bias:g} "
              f"(lambda = {1 / (1 + math.exp(-args.gate_init_bias)):.4f} at init)"
              f"   lr {lr_txt}")
    if args.min_order:
        print(f"  min order   first {args.min_order} factor(s) always on")
    if any(h_ > 0 for h_ in hard_froms):
        print(f"  hard gates  from step {{{', '.join(str(h_) if h_ else 'never' for h_ in hard_froms)}}}"
              f"  (straight-through; soft before)")
    if any(w_ > 0 for w_ in walls):
        print(f"  logit wall  c in {{{', '.join(f'{w_:g}' for w_ in walls)}}}  "
              f"(0 = off), weight {args.logit_wall_weight:g}, from step 1")
    if args.fingerprint:
        print(f"  fingerprint step 1 compared against {args.fingerprint}")
    print(f"  seeds       {', '.join(str(s) for s in seeds)}")
    if len(seeds) < 3:
        print("  WARNING: fewer than 3 seeds. Outcomes on these cells are")
        print("  bistable; a single run cannot distinguish a result from a draw.")
    print("\n  Gates are soft while training and hard at evaluation, so the")
    print("  reported cost is an integer factor count.\n")

    rows = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def emit(row):
        rows.append(row)
        cfg = {k: v for k, v in vars(args).items() if k != "ref_rows"}
        out.write_text(json.dumps({"config": cfg, "results": rows},
                                  indent=2, default=str))

    for seed in seeds:
        if "fixed" in want:
            print("-" * 78)
            print(f"fixed{args.max_order}  seed {seed}")
            print("-" * 78)
            emit(run_arm(f"fixed{args.max_order}", args.max_order, "fixed",
                         0.0, data, lut, args, seed, dlut=dlut))
        if "oracle" in want:
            print("-" * 78)
            print(f"oracle  seed {seed}")
            print("-" * 78)
            emit(run_arm("oracle", args.max_order, "oracle", 0.0,
                         data, lut, args, seed, dlut=dlut))
        if "learned" in want:
            for gamma in gammas:
                for w_ in walls:
                    for h_ in hard_froms:
                        parts = [f"g={gamma:g}"]
                        if w_:
                            parts.append(f"wall={w_:g}")
                        if h_:
                            parts.append(f"hard@{h_}")
                        tag = f"learned({','.join(parts)})"
                        print("-" * 78)
                        print(f"learned  gamma {gamma:g}  wall {w_:g}  "
                              f"hard from {h_ or 'never'}  seed {seed}")
                        print("-" * 78)
                        emit(run_arm(tag, args.max_order, "learned", gamma, data,
                                     lut, args, seed, wall=w_, hard_from=h_,
                                     dlut=dlut))

    report(rows, args)
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
