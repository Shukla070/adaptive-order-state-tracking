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
single bottleneck; both default to the previous behaviour, so passing neither
reproduces earlier runs exactly.
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

    print("\n  " + ("all self-tests passed" if ok else "SELF-TEST FAILED"))
    return ok


# ===========================================================================
# Training
# ===========================================================================

def run_arm(name, n_h, mode, gamma, data, lut, args, seed):
    """mode: 'fixed' | 'oracle' | 'learned'."""
    from exp_matched_compute import build
    from oracle_order import oracle_gates, set_external_gates

    tr_x, tr_y, te_x, te_y = data
    adaptive = mode in ("oracle", "learned")
    model = build(n_h, adaptive, args.n_heads, args.head_dim, args.hidden,
                  args.n_layers, args.vocab, seed,
                  gate_init_bias=args.gate_init_bias)
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
            bud = float(bud_t.detach().item())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % args.log_every == 0 or step == 1:
            extra = ""
            wn = None
            if mode == "learned":
                wn = gate_weight_norm(model)
                extra = (f"  gamma_eff {gw:.4f}  mean n_t {bud:.3f}"
                         f"  gate |W| {wn:.3e}")
            trace.append((step, round(task.item(), 4), round(bud, 4),
                          None if wn is None else round(wn, 6)))
            print(f"      step {step:>6}  task {task.item():.4f}{extra}")
    dt = time.perf_counter() - t0
    if mode == "oracle":
        set_external_gates(model, None)

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
                st = allocation_stats(n_t, x, lut, n_h)
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

    row = dict(arm=name, mode=mode, n_h=n_h, gamma=gamma, seed=seed,
               token_acc=tc / tt, seq_acc=sc / max(1, ns), train_s=dt,
               gate_lr=gate_lr, gate_init_bias=args.gate_init_bias,
               gate_w_final=(gate_weight_norm(model) if mode == "learned" else None),
               loss_trace=trace, alloc=alloc)
    del model
    torch.cuda.empty_cache()
    return row


def report(rows, args):
    print("=" * 78)
    print(f"{'arm':<16} {'seed':>5} {'token':>8} {'seq':>7} {'cost':>7} "
          f"{'mae':>6} {'under':>7} {'corr':>6}")
    print("-" * 78)
    for r in rows:
        a = r["alloc"]
        f = lambda v, p=3: ("  n/a" if v is None else f"{v:.{p}f}")
        print(f"{r['arm']:<16} {r['seed']:>5} {r['token_acc']:>8.4f} "
              f"{r['seq_acc']:>7.4f} {f(a['mean_n_t']):>7} {f(a['mae'],2):>6} "
              f"{f(a['under_frac'],2):>7} {f(a['corr'],2):>6}")
    print("=" * 78)

    orc = [r for r in rows if r["mode"] == "oracle"]
    if not orc:
        print("  NO VERDICT: no oracle arm in this run to compare against.")
        return
    best_orc = max(orc, key=lambda r: r["token_acc"])
    print(f"  oracle reference: token {best_orc['token_acc']:.4f} at cost "
          f"{best_orc['alloc']['mean_n_t']:.3f}\n")

    for gamma in sorted({r["gamma"] for r in rows if r["mode"] == "learned"}):
        arms = sorted([r for r in rows if r["mode"] == "learned"
                       and r["gamma"] == gamma], key=lambda r: r["seed"])
        n_closed, n_run = closure_fraction(arms)
        print(f"  gamma = {gamma:g}   gate closed on {n_closed} of {n_run} run(s)")
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
    ap.add_argument("--arms", default="fixed,oracle,learned")
    ap.add_argument("--log_every", type=int, default=2000)
    ap.add_argument("--out", default="results/gate_train.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        print("=" * 78)
        print("SELF-TEST — allocation statistics and schedule, known answers")
        print("=" * 78)
        raise SystemExit(0 if selftest() else 1)

    args.vocab = N_GROUP + N_SPECIAL
    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA")

    from exp_matched_compute import load
    from oracle_order import load_ell_lookup

    gammas = [float(x) for x in args.gammas.split(",")]
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
    data = load(csv, args.n_train, args.n_test)

    print("=" * 78)
    print("OBJECTIVE O4 — the halting gate, trained end-to-end")
    print("=" * 78)
    print(f"  data        {csv.name}   p = {args.p}")
    print(f"  demand      E[D] = {m['empirical_mean_ell']:.3f}   "
          f"fixed order must provision {m['fixed_order_cost']}")
    print(f"  budget      {args.steps} steps x batch {args.batch}, lr {args.lr}")
    print(f"  gammas      {', '.join(f'{g:g}' for g in gammas)}"
          f"   (0 = gate present, never penalised)")
    print(f"  schedule    penalty off for {args.budget_warmup} steps, then "
          f"ramped over {args.budget_ramp}")
    print(f"  gate        init bias {args.gate_init_bias:g} "
          f"(lambda = {1 / (1 + math.exp(-args.gate_init_bias)):.4f} at init)"
          f"   lr {'= model lr' if args.gate_lr is None else f'{args.gate_lr:g}'}")
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
        out.write_text(json.dumps({"config": vars(args), "results": rows},
                                  indent=2, default=str))

    for seed in seeds:
        if "fixed" in want:
            print("-" * 78)
            print(f"fixed{args.max_order}  seed {seed}")
            print("-" * 78)
            emit(run_arm(f"fixed{args.max_order}", args.max_order, "fixed",
                         0.0, data, lut, args, seed))
        if "oracle" in want:
            print("-" * 78)
            print(f"oracle  seed {seed}")
            print("-" * 78)
            emit(run_arm("oracle", args.max_order, "oracle", 0.0,
                         data, lut, args, seed))
        if "learned" in want:
            for gamma in gammas:
                print("-" * 78)
                print(f"learned  gamma {gamma:g}  seed {seed}")
                print("-" * 78)
                emit(run_arm(f"learned(g={gamma:g})", args.max_order, "learned",
                             gamma, data, lut, args, seed))

    report(rows, args)
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
