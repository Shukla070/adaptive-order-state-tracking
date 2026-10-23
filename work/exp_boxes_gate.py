"""The learned gate on Boxes: can it keep the accuracy of 2 factors per token
while using fewer, and does it give the second factor to the tokens that
change the state?

Pre-registered in PLAN.md ("Learned gate on Boxes", 22 October 2026) before
any run. Everything not named here is identical to work/exp_boxes_order.py
(Shashwat's model, data, optimiser, schedule and seed scheme, 16 000 steps).

Arms (maximum order K = 2; the order sweep found a third factor adds nothing)
  rule     a hand rule: the second factor only for tokens inside the operation
           sentences (the 'operation' kind of boxes_lib), one factor elsewhere.
           No gate; same size as fixed 2 (469,120 parameters).
  learned  the halting gate of src/gating.py with the S5 recipe: open start
           (bias 4), hard straight-through gates from step 1, first factor
           always on, budget penalty gamma * mean order over non-padding tokens,
           ramped linearly from step 1 to gamma over 4 000 steps. The gate's
           parameters are taken out of the MLP width, so the model stays at
           469,120 parameters.
Fixed 1 and fixed 2 are not re-run: they are the 5 + 5 runs of
results/boxes_order.json, made with the same code path.

Stages
  probe   rule, seeds 0-4; learned at gamma 0.01, 0.03, 0.1, seeds 0-1.
          gamma is chosen on the DEV split: the largest gamma whose median
          dev E is within 0.02 of the best gamma's median dev E.
  final   learned at the chosen gamma, seeds 10-14, reported on the test split.

E = unweighted mean exact match over 3, 4 and 5+ operations (as in the sweep).
Order = mean factors per non-padding token with hard gates, over both layers.

Decision (final stage, 5 runs), written before any run
  GATE WORKS              median E >= 0.885 and median order <= 1.6
  ACCURACY, NO SAVING     median E >= 0.885 and median order > 1.6
                          (bar amended from 1.5 on 22 Oct before any full run: on the
                          test split the hand rule itself costs 1.508)
  LOSES ACCURACY          median E < 0.885; read against the rule arm
  NOT CONVERGED           (checked first) median quick-eval E rose > 0.03 over the
                          last quarter of training: no decision, rerun longer
  Mechanism check, reported either way: operation tokens get a higher mean
  order than description tokens in at least 4 of 5 runs.
  Rule arm: median E >= 0.885 means an uneven allocation is enough here.
Restarts are not used: the adopted rule's threshold (1.20 at step 500) belongs
to the S5 allocation, and Boxes has no known target allocation. Every run logs
its order and task loss at step 500 so a Boxes signal can be defined later.

Follow-up, pre-registered 23 October after the final stage lost accuracy (PLAN.md);
amended the same day, before any follow-up run
  Scores are LATE AVERAGES: E averaged over the last 4 quick evaluations (steps 10k-16k).
  With a constant learning rate, final-step E swings by ~0.03 between evaluations; fixed
  2's final-E median (0.915) is 0.04 above its late average (0.872), so the 0.885 bar
  would fail fixed 2 itself in about 2 of 3 repeats. Bars are now relative to the fixed-2
  reference's median late average: bar = median - 0.03.
  layers21  fixed orders [2, 1]: 2 factors in the first layer, 1 in the second, the
            second layer really built with one factor. No gate. 469,120 parameters
            (MLP width 522). Seeds 0-7. Plus 3 more fixed-2 runs ('fixed22', seeds 5-7,
            identical to the sweep's fixed 2: self-test G10), so the reference has 8.
            Median late average >= bar -> [2,1] HOLDS: time it against fixed 2 next.
            Otherwise -> [2,1] LOSES ACCURACY. Mann-Whitney p against fixed 1 and
            fixed 2 is reported beside the decision, not used for it.
  probe2    the learned gate at gamma 0.001 and 0.003, seeds 0-2; a hypothesis test (is
            the penalty too strong for this task?), not a fix. gamma chosen on the DEV
            late average by the usual rule; the final stage (5 runs, seeds 10-14) runs
            only if that median reaches bar - 0.03, and is decided with late averages:
            median >= bar and order <= 1.6 -> GATE WORKS (etc.).
            Read-out: change in order over the last quarter of training (median
            over runs); "settles" = falls by at most 0.02.

Usage
  python work/exp_boxes_gate.py --selftest
  python -u work/exp_boxes_gate.py --stage probe --steps 300 --out results/boxes_gate_smoke.json   # smoke
  nohup python -u work/exp_boxes_gate.py --stage probe > results/boxes_gate_probe.log 2>&1 &
  python work/exp_boxes_gate.py --summary
  nohup python -u work/exp_boxes_gate.py --stage final --gamma <chosen> > results/boxes_gate_final.log 2>&1 &
  python -u work/exp_boxes_gate.py --stage layers21 --steps 300 --out results/boxes_followup_smoke.json   # smoke
  python -u work/exp_boxes_gate.py --stage probe2 --steps 300 --out results/boxes_followup_smoke.json     # smoke
  nohup bash work/run_boxes_followup.sh > results/boxes_followup_driver.log 2>&1 &
"""

import argparse
import copy
import json
import math
import os
import random
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "src"))
import boxes_lib as bl  # noqa: E402
import exp_boxes_order as eo  # noqa: E402
from gating import HaltingGate  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

K = 2
TARGET_PARAMS = eo.REF_PARAMS            # 469,120: Shashwat's fixed-2 Boxes model
FIXED2_MEDIAN_E, FIXED1_MEDIAN_E = 0.915, 0.810   # results/boxes_order.json
# ORDER_BAR amended 22 Oct before any full run (PLAN.md): 1.5 -> 1.6. The test split spreads
# examples evenly over 0-8 operations, so operation tokens are about half of all tokens and the
# hand rule itself costs 1.508 there; 1.6 means at least 40% of second factors skipped.
E_BAR, ORDER_BAR, SELECT_TOL = FIXED2_MEDIAN_E - 0.03, 1.6, 0.02
IN_DIST_MAX_OPS = 2           # order is also reported on test examples like the training data
PROBE_GAMMAS, PROBE_SEEDS, RULE_SEEDS, FINAL_SEEDS = (0.01, 0.03, 0.1), (0, 1), (0, 1, 2, 3, 4), (10, 11, 12, 13, 14)
RAMP, GATE_BIAS, GATE_MLP = 4000, 4.0, 64
KIND_NAMES = ["description", "operation", "query", "answer"]
OPERATION = 1
DEV_PER_BUCKET = 1000
# Follow-up, 23 October (see the module docstring and PLAN.md)
PROBE2_GAMMAS, PROBE2_SEEDS = (0.001, 0.003), (0, 1, 2)
LAYER_ORDERS = (2, 1)
FIXED_ORDERS = {"fixed21": LAYER_ORDERS, "fixed22": (2, 2)}
# Amended 23 Oct before any follow-up run (PLAN.md): 8 runs of [2, 1], and 3 more fixed-2
# runs (seeds 5-7, this code path) so the fixed-2 reference has 8 runs (5 from the sweep).
LAYER_SEEDS, REF2_SEEDS = (0, 1, 2, 3, 4, 5, 6, 7), (5, 6, 7)
LATE_ORDER_DROP = 0.02
# Steadier score, 23 Oct: E averaged over the last LATE_N quick evaluations (steps 10k-16k
# of 16k). Final-step E swings by ~0.03 between evaluations because the learning rate is
# constant; fixed 2's final-E median (0.915) sits 0.04 above its late average (0.872).
LATE_N, LATE_TOL = 4, 0.03
SWEEP_JSON = os.path.join(REPO, "results", "boxes_order.json")


# ----------------------------------------------------------------------------- model

class GatedBranch(nn.Module):
    """boxes_lib.DeltaProductBranch with each factor's beta scaled by a gate.
    Wraps the original branch (its parameters are reused unchanged), so with
    every gate at 1 the output is the original branch's, bit for bit."""

    def __init__(self, base, mode, d):
        super().__init__()
        self.base, self.mode = base, mode
        self.ext = None            # (B, T, K) gates for mode 'rule' / 'open'
        self.last_n = None         # (B, T) order per token, hard
        if mode == "learned":
            self.gate = HaltingGate(d, base.n_h, mlp_dim=GATE_MLP, init_open_bias=GATE_BIAS, min_order=1)

    def gates_for(self, x):
        B, T, _ = x.shape
        if self.mode == "learned":
            g, n = self.gate(x, hard=True)
        elif self.mode == "rule":
            g = self.ext
            n = g.sum(-1)
        else:  # open
            g = x.new_ones(B, T, self.base.n_h)
            n = g.sum(-1)
        self.last_n = n
        return g

    def forward(self, x):
        b = self.base
        B, T, _ = x.shape
        H, dh, n_h = b.H, b.dh, b.n_h
        q = b.cq(b.q(x)).reshape(B, T, H, dh).transpose(1, 2)
        k = b.ck(b.k(x)).reshape(B, T, n_h, H, dh).permute(0, 3, 1, 2, 4)
        v = b.cv(b.v(x)).reshape(B, T, n_h, H, dh).permute(0, 3, 1, 2, 4)
        beta = 2 * torch.sigmoid(b.b(x)).reshape(B, T, n_h, H).permute(0, 3, 1, 2)   # (B,H,T,n_h)
        beta = beta * self.gates_for(x).to(beta.dtype).unsqueeze(1)
        o = bl.deltaproduct(F.normalize(q, dim=-1), F.normalize(k, dim=-1), v, beta, b.chunk)
        return o.transpose(1, 2)


def branches(model):
    return [blk.mix.delta for blk in model.blocks]


def build_model(vocab, a, mode, d_ff, orders=None):
    """mode: 'learned' | 'rule' | 'open' | 'fixed21' | 'fixed22'. The LM is built exactly as
    boxes_lib builds it; gates (learned mode only) are created afterwards. 'fixed21' rebuilds
    layer i's branch with orders[i] factors (really fewer parameters and less compute,
    not factors multiplied by zero) and runs it with every gate open; 'fixed22' is fixed 2
    built the same way (the reference)."""
    an = argparse.Namespace(**{**vars(a), "n_h": K})
    m = bl.build(vocab, an, "delta", None, d_ff)
    if mode in FIXED_ORDERS:
        orders = FIXED_ORDERS[mode] if orders is None else orders
        if len(orders) != len(m.blocks):
            raise ValueError(f"orders {orders} given for {len(m.blocks)} layers")
        for blk, n in zip(m.blocks, orders):
            if n != K:
                blk.mix.delta = bl.DeltaProductBranch(an.d, an.heads, an.head_dim, n, an.chunk, an.conv_k)
            blk.mix.delta = GatedBranch(blk.mix.delta, "open", an.d)
        return m
    for blk in m.blocks:
        blk.mix.delta = GatedBranch(blk.mix.delta, mode, an.d)
    return m


def layer_orders(model):
    """Factors per layer as built (n_h of each branch)."""
    return [br.base.n_h for br in branches(model)]


def matched_d_ff(vocab, a, mode, target=TARGET_PARAMS):
    p1, p2 = bl.n_params(build_model(vocab, a, mode, 1)), bl.n_params(build_model(vocab, a, mode, 2))
    slope = p2 - p1
    return max(1, round((target - (p1 - slope)) / slope))


def rule_gates(kinds):
    """(B, T) kinds -> (B, T, K): factor 1 always, factor 2 on operation tokens."""
    g = torch.ones(*kinds.shape, K, dtype=torch.float32, device=kinds.device)
    g[..., 1] = (kinds == OPERATION).float()
    return g


def set_rule(model, kinds):
    g = rule_gates(kinds)
    for br in branches(model):
        br.ext = g


def mean_order(model, valid):
    """Mean factors per non-padding token, averaged over layers. Keeps the graph."""
    return torch.stack([br.last_n[valid].mean() for br in branches(model)]).mean()


def budget_weight(step, gamma, warmup=0, ramp=RAMP):
    """Same schedule as work/exp_gate_train.py."""
    if gamma == 0.0 or step <= warmup:
        return 0.0
    if ramp <= 0:
        return gamma
    return gamma * min(1.0, (step - warmup) / ramp)


# ----------------------------------------------------------------------------- evaluation

def dev_task(task, boxes_dir, per_bucket=DEV_PER_BUCKET, seed=1234):
    """A copy of the test task whose evaluation sets come from dev-t5.jsonl."""
    path = bl._boxes_locate(boxes_dir, "dev-t5.jsonl")
    if path is None:
        raise SystemExit(f"dev-t5.jsonl not found under {boxes_dir}")
    rng = random.Random(seed)
    by = {}
    for r in bl._read_jsonl(path):
        by.setdefault(min(r["numops"], bl.BOXES_MAX_BUCKET), []).append(r)
    dt = copy.copy(task)
    dt.eval_sets = {}
    for b in task.bucket_of_group:
        rows = by.get(b, [])
        rows = rows if len(rows) <= per_bucket else rng.sample(rows, per_bucket)
        dt.eval_sets[b] = [task.encode(r, grow=False) for r in rows]
    return dt


@torch.no_grad()
def evaluate(model, task, mode, device, n=None):
    model.eval()
    G = len(task.group_names)
    right, total = torch.zeros(G), torch.zeros(G)
    osum = torch.zeros(len(KIND_NAMES)); ocnt = torch.zeros(len(KIND_NAMES))
    gsum = torch.zeros(G); gcnt = torch.zeros(G)
    for b in task.eval_batches(n):
        bd = bl.Batch(*[t.to(device) for t in b])
        if mode == "rule":
            set_rule(model, bd.kinds)
        r, t = bl.score(model(bd.x), bd, task.metric, G)
        right, total = right + r, total + t
        nt = torch.stack([br.last_n for br in branches(model)]).mean(0).float().cpu()   # (B, T)
        for ki in range(len(KIND_NAMES)):
            sel = b.kinds == ki
            osum[ki] += nt[sel].sum(); ocnt[ki] += sel.sum()
        valid = b.kinds >= 0
        for gi in range(G):
            sel = valid & (b.group == gi)
            gsum[gi] += nt[sel].sum(); gcnt[gi] += sel.sum()
    model.train()
    acc = {name: (right[i] / total[i]).item() if total[i] > 0 else float("nan") for i, name in enumerate(task.group_names)}
    by_kind = {KIND_NAMES[i]: (osum[i] / ocnt[i]).item() for i in range(len(KIND_NAMES)) if ocnt[i] > 0}
    order = (osum.sum() / ocnt.sum()).item()
    by_group = {name: (gsum[i] / gcnt[i]).item() for i, name in enumerate(task.group_names) if gcnt[i] > 0}
    return acc, order, by_kind, by_group


# ----------------------------------------------------------------------------- training

def run(a, task, dtask, mode, gamma, seed, d_ff, device, steps, bs, log=print, save_path=None):
    """Mirrors boxes_lib.run_one (same seed handling, optimiser, schedule, clipping,
    batch generator), plus the gate's budget penalty and the rule's gates."""
    bl.set_seed(seed)
    model = build_model(task.vocab_size, a, mode, d_ff).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd)

    def lr_at(s):
        w = min(1.0, (s + 1) / a.warmup)
        return w * (0.5 * (1 + math.cos(math.pi * s / steps)) if a.lr_schedule == "cosine" else 1.0)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    gen = torch.Generator().manual_seed(seed * 7919 + 1)
    t0, curve_loss, curve_eval, run_l, run_o, diverged = time.time(), [], [], [], [], False
    curve_order, curve_dev = [], []
    at500 = None
    for step in range(1, steps + 1):
        b = bl.Batch(*[t.to(device) for t in task.train_batch(bs, gen)])
        if mode == "rule":
            set_rule(model, b.kinds)
        tl = F.cross_entropy(model(b.x)[b.mask], b.y[b.mask])
        order = mean_order(model, b.kinds >= 0)
        gw = budget_weight(step, gamma) if mode == "learned" else 0.0
        loss = tl + gw * order if gw > 0 else tl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if not math.isfinite(loss.item()):
            diverged = True
            log(f"  [{mode} g{gamma} seed{seed}] loss not finite at step {step}; stopping")
            break
        run_l.append(tl.item()); run_o.append(order.item())
        if step == 500:
            at500 = {"order": order.item(), "task_loss": tl.item()}
        if step % a.log_every == 0:
            curve_loss.append((step, sum(run_l) / len(run_l), sum(run_o) / len(run_o)))
            log(f"  [{mode} g{gamma} seed{seed}] step {step:6d}  task {curve_loss[-1][1]:.4f}  order {curve_loss[-1][2]:.3f}"
                f"  gamma_eff {gw:.4f}  {(time.time() - t0) / 60:.1f} min")
            run_l, run_o = [], []
        if a.eval_every and step % a.eval_every == 0:
            acc, o, _, _ = evaluate(model, task, mode, device, n=a.quick_n)
            curve_eval.append((step, acc))
            curve_order.append((step, o))
            msg = f"  [{mode} g{gamma} seed{seed}]   quick-eval E {eo.extrap_score(acc):.3f}  order {o:.3f}" if has_extrap(acc) else ""
            if dtask is not None and step > steps - LATE_N * a.eval_every:   # dev at the evaluations the late average uses
                dacc = evaluate(model, dtask, mode, device, n=a.quick_n)[0]
                curve_dev.append((step, dacc))
                msg += f"  dev E {eo.extrap_score(dacc):.3f}" if has_extrap(dacc) else ""
            if msg:
                log(msg)
    final, order, by_kind, by_group = evaluate(model, task, mode, device)
    dev, dev_order, _, _ = evaluate(model, dtask, mode, device) if dtask is not None else ({}, None, None, None)
    buckets = getattr(task, "bucket_of_group", None) or []
    in_dist = [g for g, bk in zip(task.group_names, buckets) if bk <= IN_DIST_MAX_OPS and g in by_group]
    order_in = statistics.mean(by_group[g] for g in in_dist) if in_dist else None
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "mode": mode, "gamma": gamma, "seed": seed, "d_ff": d_ff,
                    "build": {k: getattr(a, k) for k in ("d", "layers", "heads", "head_dim", "chunk", "conv_k")},
                    "vocab_size": task.vocab_size, "K": K, "layer_orders": layer_orders(model)}, save_path)
    return {"mode": mode, "gamma": gamma, "seed": seed, "params": bl.n_params(model), "d_ff": d_ff, "steps": step,
            "diverged": diverged, "minutes": (time.time() - t0) / 60, "final": final,
            "extrap": eo.extrap_score(final) if all(c in final for c in eo.EXTRAP) else None, "order": order, "order_by_kind": by_kind,
            "order_by_group": by_group, "order_in_dist": order_in, "checkpoint": save_path,
            "dev": dev, "dev_extrap": eo.extrap_score(dev) if dev else None, "at500": at500,
            "curve_loss": curve_loss, "curve_eval": curve_eval, "curve_order": curve_order, "curve_dev": curve_dev,
            "late_E": late_mean(curve_eval), "dev_late_E": late_mean(curve_dev),
            "layer_orders": layer_orders(model)}


# ----------------------------------------------------------------------------- decisions

def has_extrap(acc):
    return all(c in acc for c in eo.EXTRAP)


def late_mean(curve, n=LATE_N):
    """Mean E over the last n quick evaluations of one run; None if fewer than n.
    curve: [(step, acc dict)] (lists after a JSON round trip)."""
    if len(curve) < n or not all(has_extrap(acc) for _, acc in curve[-n:]):
        return None
    return sum(eo.extrap_score(acc) for _, acc in curve[-n:]) / n


def mwu_p(a, b):
    """Exact two-sided Mann-Whitney p (ties count 1/2) by enumerating every split of the
    pooled values. Reported beside a decision, never used to make one."""
    import itertools
    pooled, na = list(a) + list(b), len(a)

    def U(ia):
        sa = set(ia)
        xa = [pooled[i] for i in ia]
        xb = [pooled[i] for i in range(len(pooled)) if i not in sa]
        return sum((x > y) + 0.5 * (x == y) for x in xa for y in xb)

    u0 = U(range(na))
    us = [U(c) for c in itertools.combinations(range(len(pooled)), na)]
    lo = sum(u <= u0 + 1e-12 for u in us) / len(us)
    hi = sum(u >= u0 - 1e-12 for u in us) / len(us)
    return min(1.0, 2 * min(lo, hi))


def references(path=SWEEP_JSON, runs=()):
    """Late averages of the references: fixed 1 and fixed 2 from the order sweep, plus
    this script's fixed-2 runs ('fixed22', same code path; self-test G10)."""
    ref = {1: [], 2: []}
    if os.path.exists(path):
        with open(path) as f:
            for k, r in json.load(f)["runs"].items():
                nh = int(k.split("|nh")[1].split("|")[0])
                if nh in ref and late_mean(r["curve_eval"]) is not None:
                    ref[nh].append(late_mean(r["curve_eval"]))
    ref[2] += [r["late_E"] for r in runs if r["mode"] == "fixed22" and r.get("late_E") is not None]
    return ref


def late_bar(ref2):
    return statistics.median(ref2) - LATE_TOL


def choose_gamma(probe):
    """probe: {gamma: [dev E per run]} -> the largest gamma whose median dev E is within
    SELECT_TOL of the best gamma's median."""
    med = {g: statistics.median(v) for g, v in probe.items()}
    best = max(med.values())
    return max(g for g, m in med.items() if m >= best - SELECT_TOL)


def decide_gate(Es, orders):
    mE, mO = statistics.median(Es), statistics.median(orders)
    if mE >= E_BAR and mO <= ORDER_BAR:
        return "GATE WORKS"
    if mE >= E_BAR:
        return "ACCURACY, NO SAVING"
    return "LOSES ACCURACY"


def mechanism_ok(by_kinds):
    """At least 4 of 5 runs give operation tokens a higher order than description tokens."""
    hits = sum(bk["operation"] > bk["description"] for bk in by_kinds)
    return hits >= 4, hits


def decide_layers(late21, ref2):
    """Follow-up arm layers21 (amended 23 Oct): [2, 1] holds if its median late average is at
    least the fixed-2 reference's median late average minus LATE_TOL."""
    return "[2,1] HOLDS" if statistics.median(late21) >= late_bar(ref2) - 1e-12 else "[2,1] LOSES ACCURACY"


def late_order_change(curve_order, steps):
    """Order at the last quick evaluation minus order at the last evaluation at or
    before 3/4 of training (one run). None if there is no such pair."""
    early = [(s, o) for s, o in curve_order if s <= 0.75 * steps]
    if not early or not curve_order or curve_order[-1][0] == early[-1][0]:
        return None
    return curve_order[-1][1] - early[-1][1]


def order_settles(curves, steps):
    """Median over runs of the late order change, and whether it falls by at most
    LATE_ORDER_DROP. curves: list of curve_order lists."""
    ch = [c for c in (late_order_change(cv, steps) for cv in curves) if c is not None]
    if not ch:
        return None, None
    m = statistics.median(ch)
    return m, m >= -LATE_ORDER_DROP - 1e-9   # inclusive; tolerance for float rounding of the difference


def probe_seeds(gamma):
    return PROBE2_SEEDS if gamma in PROBE2_GAMMAS else PROBE_SEEDS


def followup_gamma(runs):
    """gamma chosen on the dev late average among PROBE2_GAMMAS, once every probe2 run
    exists; else None. Returns (gamma, its median dev late average)."""
    probe = {}
    for g in PROBE2_GAMMAS:
        rs = [r for r in runs if r["mode"] == "learned" and r["gamma"] == g and r["seed"] in PROBE2_SEEDS
              and r.get("dev_late_E") is not None]
        if len(rs) < len(PROBE2_SEEDS):
            return None
        probe[g] = [r["dev_late_E"] for r in rs]
    g = choose_gamma(probe)
    return g, statistics.median(probe[g])


def decide_gate_late(lateEs, orders, ref2):
    """Final-stage rule for the follow-up gammas: as decide_gate, with late averages and the
    late bar in place of final-step E and 0.885."""
    mE, mO = statistics.median(lateEs), statistics.median(orders)
    if mE >= late_bar(ref2) - 1e-12:
        return "GATE WORKS" if mO <= ORDER_BAR else "ACCURACY, NO SAVING"
    return "LOSES ACCURACY"


def final_allowed(dev_median, ref2):
    """The follow-up final stage runs only if the chosen gamma's median dev late average
    reaches the late bar minus LATE_TOL."""
    return dev_median >= late_bar(ref2) - LATE_TOL - 1e-12


# ----------------------------------------------------------------------------- self-test

def selftest(device):
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}", flush=True)

    torch.manual_seed(0)
    print("G1  gated branch against the original branch (float64)")
    dt = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    base = bl.DeltaProductBranch(32, 2, 16, K, 16, 4)
    x = torch.randn(3, 40, 32)
    ref = base(x)
    gb = GatedBranch(base, "open", 32)
    check("all gates open = the original branch, exactly", (gb(x) - ref).abs().max().item() == 0.0)
    gr = GatedBranch(base, "rule", 32)
    gr.ext = torch.ones(3, 40, K)
    check("rule gates all 1 = the original branch, exactly", (gr(x) - ref).abs().max().item() == 0.0)
    # factor 2 closed everywhere == a recurrence that uses factor 1 only
    gr.ext = torch.ones(3, 40, K); gr.ext[..., 1] = 0
    B, T = 3, 40
    q = base.cq(base.q(x)).reshape(B, T, 2, 16).transpose(1, 2)
    k = base.ck(base.k(x)).reshape(B, T, K, 2, 16).permute(0, 3, 1, 2, 4)
    v = base.cv(base.v(x)).reshape(B, T, K, 2, 16).permute(0, 3, 1, 2, 4)
    beta = 2 * torch.sigmoid(base.b(x)).reshape(B, T, K, 2).permute(0, 3, 1, 2)
    one = bl.deltaproduct_reference(F.normalize(q, dim=-1), F.normalize(k, dim=-1)[..., :1, :], v[..., :1, :], beta[..., :1]).transpose(1, 2)
    err = (gr(x) - one).abs().max().item()
    check("second factor closed = recurrence with the first factor only", err < 1e-10, f"max err {err:.1e}")
    check("control: with the second factor open the output differs from that", (ref - one).abs().max().item() > 1e-3)
    # closing on some tokens only: token t uses 1 factor, others 2
    g = torch.ones(3, 40, K); g[:, ::3, 1] = 0
    gr.ext = g
    S = torch.zeros(B, 2, 16, 16); outs = []
    kn, qn = F.normalize(k, dim=-1), F.normalize(q, dim=-1)
    for t in range(T):
        for i in range(K):
            bt = (beta[:, :, t, i] * g[:, t, i, None])[..., None, None]
            kt, vt = kn[:, :, t, i], v[:, :, t, i]
            S = S - bt * torch.einsum("bhk,bhv->bhkv", kt, torch.einsum("bhk,bhkv->bhv", kt, S)) + bt * torch.einsum("bhk,bhv->bhkv", kt, vt)
        outs.append(torch.einsum("bhk,bhkv->bhv", qn[:, :, t], S))
    mixed = torch.stack(outs, 2).transpose(1, 2)
    err = (gr(x) - mixed).abs().max().item()
    check("per-token closing = an explicit per-token loop", err < 1e-10, f"max err {err:.1e}")
    torch.set_default_dtype(dt)

    print("G2  the learned gate in this model")
    lg = GatedBranch(bl.DeltaProductBranch(32, 2, 16, K, 16, 4), "learned", 32)
    xg = torch.randn(2, 10, 32)
    lg(xg)
    check("open start: every token at order 2 (hard)", torch.all(lg.last_n == 2).item())
    with torch.no_grad():
        lg.gate.mlp[-1].bias.fill_(-6.0)
    lg(xg)
    check("closed gate: order 1, never 0 (first factor always on)", torch.all(lg.last_n == 1).item())
    with torch.no_grad():
        lg.gate.mlp[-1].bias.fill_(0.3)
    out = lg(xg)
    out.sum().backward()
    check("hard gates still pass gradient to the gate", lg.gate.mlp[-1].bias.grad is not None and lg.gate.mlp[-1].bias.grad.abs().sum().item() > 0)

    print("G3  rule gates and the order measure")
    kinds = torch.tensor([[0, 0, 1, 1, 1, 2, 3, -1], [0, 1, 2, 3, -1, -1, -1, -1]])
    rg = rule_gates(kinds)
    check("operation tokens get 2 factors, all others 1 (padding too)",
          rg.sum(-1).tolist() == [[1, 1, 2, 2, 2, 1, 1, 1], [1, 2, 1, 1, 1, 1, 1, 1]])
    m = build_model(40, bl.make_args(d=32, heads=2, head_dim=16, layers=2), "rule", 32)
    set_rule(m, kinds)
    m(torch.randint(0, 40, kinds.shape))
    mo = mean_order(m, kinds >= 0).item()
    check("mean order counts non-padding tokens only: 15/11 here", abs(mo - 15 / 11) < 1e-6, f"{mo:.4f}")
    check("control: counting padding too would give 20/16, a different number", abs(15 / 11 - 20 / 16) > 0.1)

    print("G4  training loop = boxes_lib.run_one when every gate is open")
    a = bl.make_args(d=32, heads=2, head_dim=16, layers=1, n_h=K, log_every=1, eval_every=0, warmup=2)
    task = bl.RecallTask(n_keys=16, n_vals=16, train_pairs=4, eval_pairs=(4,), n_eval=8)
    ref_run = bl.run_one(a, task, "delta", None, 32, 3, "cpu", 4, 8)
    ours = run(a, task, None, "open", 0.0, 3, 32, "cpu", 4, 8, log=lambda *_: None)
    refl = [round(v, 9) for _, v in ref_run["curve_loss"]]
    ourl = [round(v, 9) for _, v, _o in ours["curve_loss"]]
    check("run() reproduces run_one's losses for 4 steps (same init, data, optimiser)", refl == ourl, f"{refl} vs {ourl}")

    print("G5  model size")
    ab = bl.make_args()
    vocab = 123
    f_rule, f_learn = matched_d_ff(vocab, ab, "rule"), matched_d_ff(vocab, ab, "learned")
    p_rule = bl.n_params(build_model(vocab, ab, "rule", f_rule))
    p_learn = bl.n_params(build_model(vocab, ab, "learned", f_learn))
    check("rule arm: d_ff 455, 469,120 parameters (= fixed 2)", f_rule == 455 and p_rule == TARGET_PARAMS, f"{f_rule}, {p_rule:,}")
    check("learned arm within 0.2% of 469,120", abs(p_learn - TARGET_PARAMS) / TARGET_PARAMS < 0.002, f"d_ff {f_learn}, {p_learn:,}")
    p_naive = bl.n_params(build_model(vocab, ab, "learned", 455))
    check("control: the gate at d_ff 455 would add > 3% parameters", (p_naive - TARGET_PARAMS) / TARGET_PARAMS > 0.03, f"{p_naive:,}")

    print("G6  penalty schedule, gamma choice and decisions on known answers")
    check("ramp: 0 at step 0, gamma/2 at 2 000, gamma from 4 000", budget_weight(0, 0.1) == 0 and abs(budget_weight(2000, 0.1) - 0.05) < 1e-12
          and budget_weight(4000, 0.1) == 0.1 and budget_weight(9000, 0.1) == 0.1)
    check("gamma choice: the largest within 0.02 of the best", choose_gamma({0.01: [0.92, 0.90], 0.03: [0.905, 0.91], 0.1: [0.84, 0.86]}) == 0.03)
    check("gamma choice: if the largest is best, take it", choose_gamma({0.01: [0.80, 0.80], 0.03: [0.85, 0.85], 0.1: [0.90, 0.90]}) == 0.1)
    check("control: choosing by best dev E alone would pick 0.01 there",
          max({0.01: 0.91, 0.03: 0.9075}.items(), key=lambda kv: kv[1])[0] == 0.01)
    check("GATE WORKS", decide_gate([0.90] * 5, [1.2] * 5) == "GATE WORKS")
    check("ACCURACY, NO SAVING", decide_gate([0.92] * 5, [1.8] * 5) == "ACCURACY, NO SAVING")
    check("LOSES ACCURACY", decide_gate([0.85] * 5, [1.1] * 5) == "LOSES ACCURACY")
    check("bar is 0.885 inclusive", decide_gate([0.885] * 5, [1.5] * 5) == "GATE WORKS" and decide_gate([0.8849] * 5, [1.5] * 5) == "LOSES ACCURACY")
    check("order bar is 1.6 inclusive (amended from 1.5)", decide_gate([0.9] * 5, [1.6] * 5) == "GATE WORKS"
          and decide_gate([0.9] * 5, [1.61] * 5) == "ACCURACY, NO SAVING")
    bk = [{"operation": 1.6, "description": 1.1}] * 4 + [{"operation": 1.0, "description": 1.2}]
    check("mechanism: 4 of 5 runs is enough, 3 of 5 is not",
          mechanism_ok(bk) == (True, 4) and mechanism_ok(bk[:3] + bk[-1:] * 2) == (False, 3))

    print("G7  a short learned run trains and reports orders by kind")
    tk = FakeBoxes()
    ab2 = bl.make_args(d=32, heads=2, head_dim=16, layers=1, log_every=1000, eval_every=0, warmup=2)
    r = run(ab2, tk, tk, "learned", 0.1, 0, 32, "cpu", 6, 4, log=lambda *_: None)
    check("runs, finite, order between 1 and 2, kinds reported, JSON-serialisable",
          not r["diverged"] and 1.0 <= r["order"] <= 2.0 and set(r["order_by_kind"]) <= set(KIND_NAMES)
          and json.loads(json.dumps(r))["steps"] == 6, f"order {r['order']:.3f}")
    r2 = run(ab2, tk, None, "rule", 0.0, 0, 32, "cpu", 3, 4, log=lambda *_: None)
    check("rule run: operation tokens at 2, description at 1",
          abs(r2["order_by_kind"].get("operation", 0) - 2) < 1e-9 and abs(r2["order_by_kind"].get("description", 0) - 1) < 1e-9, str(r2["order_by_kind"]))
    check("rule run: order 1.3 in every bucket and on 0-2 operations (3 of 10 tokens are operations)",
          all(abs(v - 1.3) < 1e-6 for v in r2["order_by_group"].values()) and abs(r2["order_in_dist"] - 1.3) < 1e-6,
          str(r2["order_by_group"]))
    import tempfile
    tmp = os.path.join(tempfile.mkdtemp(), "m.pt")
    r3 = run(ab2, tk, None, "learned", 0.1, 0, 32, "cpu", 3, 4, log=lambda *_: None, save_path=tmp)
    ck = torch.load(tmp)
    m2 = build_model(tk.vocab_size, ab2, "learned", 32)
    m2.load_state_dict(ck["state_dict"])
    xb = tk.train_batch(4, torch.Generator().manual_seed(5))
    m2.eval()
    a1 = m2(xb.x)
    bl.set_seed(123)
    m3 = build_model(tk.vocab_size, ab2, "learned", 32); m3.load_state_dict(ck["state_dict"]); m3.eval()
    check("saved model reloads and gives the same output; checkpoint path recorded",
          r3["checkpoint"] == tmp and torch.equal(a1, m3(xb.x)) and ck["d_ff"] == 32 and ck["mode"] == "learned")
    print("G8  fixed orders [2, 1] (follow-up, 23 Oct)")
    small = bl.make_args(d=32, heads=2, head_dim=16, layers=2)
    mf = build_model(40, small, "fixed21", 32)
    check("layer orders as built: [2, 1]", layer_orders(mf) == [2, 1], str(layer_orders(mf)))
    check("control: orders (2, 2) build [2, 2]", layer_orders(build_model(40, small, "fixed21", 32, orders=(2, 2))) == [2, 2])
    k1 = branches(mf)[1].base.k.weight.shape[0]
    check("second layer really has one factor's projections (k: H*dh rows, not 2*H*dh)", k1 == 2 * 16, f"{k1} rows")
    kinds = torch.tensor([[0, 1, 2, 3, -1]])
    mf(torch.randint(0, 40, kinds.shape))
    mo = mean_order(mf, kinds >= 0).item()
    check("order reported for [2, 1] is 1.5 per token", abs(mo - 1.5) < 1e-9, f"{mo}")
    dt = torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
    b1 = bl.DeltaProductBranch(32, 2, 16, 1, 16, 4)
    xx = torch.randn(2, 20, 32)
    check("a one-factor branch run with open gates = the plain one-factor branch, exactly",
          (GatedBranch(b1, "open", 32)(xx) - b1(xx)).abs().max().item() == 0.0)
    torch.set_default_dtype(dt)
    f21 = matched_d_ff(vocab, ab, "fixed21")
    p21 = bl.n_params(build_model(vocab, ab, "fixed21", f21))
    check("[2, 1] at matched width within 0.2% of 469,120", abs(p21 - TARGET_PARAMS) / TARGET_PARAMS < 0.002, f"d_ff {f21}, {p21:,}")
    check("control: at the same width [2, 1] has fewer parameters than [2, 2]",
          bl.n_params(build_model(vocab, ab, "fixed21", f21)) < bl.n_params(build_model(vocab, ab, "fixed21", f21, orders=(2, 2))))
    ab3 = bl.make_args(d=32, heads=2, head_dim=16, layers=2, log_every=1000, eval_every=2, warmup=2)
    tmp2 = os.path.join(tempfile.mkdtemp(), "f.pt")
    r4 = run(ab3, tk, None, "fixed21", 0.0, 0, 32, "cpu", 4, 4, log=lambda *_: None, save_path=tmp2)
    check("fixed21 run: order 1.5 for every kind, layer orders recorded, quick-eval orders recorded",
          all(abs(v - 1.5) < 1e-6 for v in r4["order_by_kind"].values()) and r4["layer_orders"] == [2, 1]
          and [s for s, _ in r4["curve_order"]] == [2, 4], f"{r4['order_by_kind']} {r4['curve_order']}")
    ck2 = torch.load(tmp2)
    m4 = build_model(tk.vocab_size, ab3, "fixed21", 32); m4.load_state_dict(ck2["state_dict"]); m4.eval()
    m5 = build_model(tk.vocab_size, ab3, "fixed21", 32); m5.load_state_dict(ck2["state_dict"]); m5.eval()
    check("fixed21 checkpoint reloads with layer orders [2, 1] and the same output",
          ck2["layer_orders"] == [2, 1] and torch.equal(m4(xb.x), m5(xb.x)))

    print("G9  follow-up decisions and read-outs on known answers")
    ref2_ = [0.853, 0.856, 0.872, 0.895, 0.902]          # fixed 2's late averages (sweep)
    check("late bar = median fixed-2 late average - 0.03 (0.842 here)", abs(late_bar(ref2_) - 0.842) < 1e-12)
    check("[2,1] bar inclusive", decide_layers([0.842] * 8, ref2_) == "[2,1] HOLDS" and decide_layers([0.8419] * 8, ref2_) == "[2,1] LOSES ACCURACY")
    check("[2,1] decision uses the median", decide_layers([0.5, 0.6, 0.7, 0.85, 0.86, 0.9, 0.95, 0.99], ref2_) == "[2,1] HOLDS")
    check("control: the old fixed bar 0.885 would fail fixed 2's own late averages", statistics.median(ref2_) < E_BAR)
    check("gate rule with late averages: works / no saving / loses",
          decide_gate_late([0.85] * 5, [1.2] * 5, ref2_) == "GATE WORKS" and decide_gate_late([0.85] * 5, [1.7] * 5, ref2_) == "ACCURACY, NO SAVING"
          and decide_gate_late([0.83] * 5, [1.2] * 5, ref2_) == "LOSES ACCURACY")
    check("final stage allowed at dev late >= bar - 0.03 (0.812 here)", final_allowed(0.812, ref2_) and not final_allowed(0.8119, ref2_))
    f1_ = [0.614, 0.713, 0.799, 0.833, 0.881]
    p_ = mwu_p(f1_, ref2_)
    check("Mann-Whitney exact p: fixed 1 vs fixed 2 late averages = 2 x 7/252 = 0.0556", abs(p_ - 14 / 252) < 1e-9, f"{p_:.4f}")
    check("Mann-Whitney: identical groups give p = 1", mwu_p([1, 2, 3], [1, 2, 3]) == 1.0)
    acc_ = lambda e: {"ops=3": e, "ops=4": e, "ops>=5": e}
    cur = [(s, acc_(e)) for s, e in zip(range(2000, 18000, 2000), (0.1, 0.2, 0.3, 0.4, 0.8, 0.9, 0.7, 0.6))]
    check("late average = mean of the last 4 quick evaluations (0.75)", abs(late_mean(cur) - 0.75) < 1e-12, f"{late_mean(cur)}")
    check("late average: None with fewer than 4 evaluations", late_mean(cur[:3]) is None)
    check("late average survives a JSON round trip", abs(late_mean(json.loads(json.dumps(cur))) - 0.75) < 1e-12)
    cv = [(2000, 1.5), (4000, 1.4), (12000, 1.3), (14000, 1.25), (16000, 1.2)]
    ch = late_order_change(cv, 16000)
    check("late order change: last eval minus the last eval at or before 3/4 (1.2 - 1.3)", abs(ch + 0.1) < 1e-12, f"{ch}")
    check("late order change: None without an evaluation after 3/4", late_order_change([(2000, 1.5), (12000, 1.3)], 16000) is None)
    mk = lambda d: [(12000, 1.5), (16000, 1.5 + d)]
    m1_, ok1 = order_settles([mk(-0.01), mk(-0.02), mk(-0.05)], 16000)
    m2_, ok2 = order_settles([mk(-0.03), mk(-0.03), mk(0.0)], 16000)
    check("settles: median drop 0.02 counts (inclusive), 0.03 does not", ok1 is True and ok2 is False, f"{m1_:+.3f} {m2_:+.3f}")
    check("follow-up gammas use 3 seeds, the first probe 2", probe_seeds(0.001) == (0, 1, 2) and probe_seeds(0.1) == (0, 1))
    check("budget ramp reaches small gammas: 0.001 from step 4 000", budget_weight(4000, 0.001) == 0.001)
    fr = lambda g, s, e: {"mode": "learned", "gamma": g, "seed": s, "dev_late_E": e, "dev_extrap": 0.0}
    runs_ = [fr(0.001, s, e) for s, e in zip((0, 1, 2), (0.90, 0.91, 0.92))] + [fr(0.003, s, e) for s, e in zip((0, 1, 2), (0.89, 0.90, 0.905))]
    check("follow-up gamma: largest within 0.02 of the best median dev late average (0.003 here, median 0.90)",
          followup_gamma(runs_) == (0.003, 0.90))
    check("follow-up gamma: no choice until every probe2 run exists", followup_gamma(runs_[:-1]) is None)
    check("follow-up gamma: earlier probe gammas are not candidates",
          followup_gamma(runs_ + [fr(0.1, s, 0.99) for s in (0, 1, 2)]) == (0.003, 0.90))

    print("G10 the fixed-2 reference built here = the sweep's fixed 2; the run records late averages")
    bl.set_seed(7); ma = bl.build(40, bl.make_args(d=32, heads=2, head_dim=16, layers=2, n_h=K), "delta", None, 32)
    bl.set_seed(7); mb = build_model(40, small, "fixed22", 32)
    sa, sb = ma.state_dict(), {k.replace(".base.", "."): v for k, v in mb.state_dict().items()}
    check("fixed22: same parameters, same initial values as boxes_lib's fixed 2", sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa))
    xq = torch.randint(0, 40, (2, 9))
    check("fixed22: same output as boxes_lib's fixed 2", torch.equal(ma(xq), mb(xq)))
    check("fixed22 at the real size: d_ff 455, 469,120 parameters",
          matched_d_ff(vocab, ab, "fixed22") == 455 and bl.n_params(build_model(vocab, ab, "fixed22", 455)) == TARGET_PARAMS)
    ab4 = bl.make_args(d=32, heads=2, head_dim=16, layers=2, log_every=1000, eval_every=1, warmup=2)
    r5 = run(ab4, tk, tk, "fixed22", 0.0, 0, 32, "cpu", 6, 4, log=lambda *_: None)
    lm_ = late_mean(r5["curve_eval"])
    check("run records the test late average and dev at the last 4 evaluations only",
          r5["late_E"] is not None and abs(r5["late_E"] - lm_) < 1e-12 and [s for s, _ in r5["curve_dev"]] == [3, 4, 5, 6]
          and r5["dev_late_E"] is not None, f"dev steps {[s for s, _ in r5['curve_dev']]}")
    cv4 = [[s, {"ops=3": e, "ops=4": e, "ops>=5": e}] for s, e in zip(range(2000, 18000, 2000), (0, 0, 0, 0, 0.8, 0.8, 0.8, 0.8))]
    sweep = {"runs": {f"boxes|delta|nh{n}|seed{s}": {"curve_eval": cv4} for n in (1, 2, 3) for s in (0, 1)}}
    tmpj = os.path.join(tempfile.mkdtemp(), "sweep.json")
    with open(tmpj, "w") as f:
        json.dump(sweep, f)
    rf = references(tmpj, runs=[{"mode": "fixed22", "late_E": 0.7}, {"mode": "fixed21", "late_E": 0.1}, {"mode": "learned", "late_E": 0.2}])
    check("references: sweep fixed 1 and fixed 2 by n_h (fixed 3 left out), plus this script's fixed22 runs only",
          sorted(round(x, 9) for x in rf[1]) == [0.8, 0.8] and sorted(round(x, 9) for x in rf[2]) == [0.7, 0.8, 0.8], str(rf))
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return ok


class FakeBoxes:
    """A tiny Boxes-like task with all four token kinds, for the self-test."""
    metric = "exact"
    group_names = ["ops=0", "ops=1", "ops=2", "ops=3", "ops=4", "ops>=5"]
    bucket_of_group = [0, 1, 2, 3, 4, 5]
    vocab_size = 20

    def _batch(self, bs, gen):
        L = 12
        x = torch.randint(2, 20, (bs, L), generator=gen)
        y = torch.randint(2, 20, (bs, L), generator=gen)
        kinds = torch.tensor([0, 0, 0, 1, 1, 1, 2, 3, 3, 3, -1, -1]).expand(bs, L).clone()
        mask = kinds == 3
        grp = torch.randint(0, 6, (bs, 1), generator=gen).expand(bs, L).clone()
        return bl.Batch(x, y, mask, kinds, grp)

    def train_batch(self, bs, gen):
        return self._batch(bs, gen)

    def eval_batches(self, n=None, bs=8):
        gen = torch.Generator().manual_seed(1)
        for _ in range(2):
            yield self._batch(bs, gen)


# ----------------------------------------------------------------------------- driver

def load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"runs": {}}


def key(mode, gamma, seed):
    return f"boxes_gate|{mode}|g{gamma}|seed{seed}"


def summary(res):
    runs = list(res["runs"].values())
    if not runs:
        print("no runs saved yet"); return
    print(f"\n{'=' * 78}\nLEARNED GATE ON BOXES  (E = exact match on 3+ operations; order = factors per token)\n{'=' * 78}")
    print(f"  references (results/boxes_order.json): fixed 1 median E {FIXED1_MEDIAN_E}, fixed 2 median E {FIXED2_MEDIAN_E}")
    rule = [r for r in runs if r["mode"] == "rule"]
    if rule:
        Es = sorted(r["extrap"] for r in rule)
        print(f"\n  hand rule ({len(rule)} runs): E [{' '.join(f'{e:.3f}' for e in Es)}] median {statistics.median(Es):.3f}, "
              f"order {statistics.median(r['order'] for r in rule):.3f}"
              + ("" if len(rule) < 5 else f"  -> {'uneven allocation is enough' if statistics.median(Es) >= E_BAR else 'below the bar'}"))
    gammas = sorted({r["gamma"] for r in runs if r["mode"] == "learned"})
    probe = {}
    for g in gammas:
        rs = [r for r in runs if r["mode"] == "learned" and r["gamma"] == g and r["seed"] in probe_seeds(g)]
        fin = [r for r in runs if r["mode"] == "learned" and r["gamma"] == g and r["seed"] in FINAL_SEEDS]
        for label, group in (("probe", rs), ("final", fin)):
            if not group:
                continue
            print(f"\n  learned gate, gamma {g}, {label} ({len(group)} runs)")
            for r in sorted(group, key=lambda r: r["seed"]):
                bk = "  ".join(f"{k} {v:.2f}" for k, v in r["order_by_kind"].items())
                dv = f"dev E {r['dev_extrap']:.3f}  " if r.get("dev_extrap") is not None else ""
                a5 = r.get("at500") or {}
                oi = r.get("order_in_dist")
                oi = f" (0-2 ops {oi:.3f})" if oi is not None else ""
                lm = late_mean(r.get("curve_eval") or [])
                lm = f"(late {lm:.3f})  " if lm is not None else ""
                dl = f"(dev late {r['dev_late_E']:.3f})  " if r.get("dev_late_E") is not None else ""
                print(f"     seed {r['seed']:>2}: test E {r['extrap']:.3f} {lm}{dv}{dl}order {r['order']:.3f}{oi}  [{bk}]"
                      f"  step-500 order {a5.get('order', float('nan')):.3f}")
        if g in PROBE2_GAMMAS and rs:
            m, ok = order_settles([r.get("curve_order") or [] for r in rs], rs[0]["steps"])
            if m is not None:
                print(f"     order change over the last quarter, median {m:+.3f} -> {'settles' if ok else 'still falling'}")
        if g in PROBE_GAMMAS and rs and all(r.get("dev_extrap") is not None for r in rs):
            probe[g] = [r["dev_extrap"] for r in rs]
    if probe and all(len(v) == len(PROBE_SEEDS) for v in probe.values()) and len(probe) == len(PROBE_GAMMAS):
        print(f"\n  GAMMA CHOSEN ON DEV (largest within {SELECT_TOL} of the best median): {choose_gamma(probe)}")
    ref = references(runs=runs)
    fmt = lambda xs: "[" + " ".join(f"{x:.3f}" for x in sorted(xs)) + "]"
    if ref[2]:
        print(f"\n  FOLLOW-UP (amended 23 Oct): scores are late averages (mean E over the last {LATE_N} quick evaluations)")
        print(f"     fixed 1 ({len(ref[1])} runs) {fmt(ref[1])} median {statistics.median(ref[1]):.3f}")
        print(f"     fixed 2 ({len(ref[2])} runs, sweep + fixed22) {fmt(ref[2])} median {statistics.median(ref[2]):.3f}"
              f"  -> bar {late_bar(ref[2]):.3f}")
    g2 = followup_gamma(runs)
    if g2 is not None and ref[2]:
        print(f"  FOLLOW-UP GAMMA CHOSEN ON DEV LATE AVERAGE among {PROBE2_GAMMAS}: {g2[0]} (median dev late {g2[1]:.3f})"
              f"  -> final stage {'runs' if final_allowed(g2[1], ref[2]) else 'not run: below bar - ' + str(LATE_TOL)}")
    fx = [r for r in runs if r["mode"] == "fixed21" and r.get("late_E") is not None]
    if fx and ref[2]:
        L = [r["late_E"] for r in fx]
        done = len(fx) >= len(LAYER_SEEDS) and len([r for r in runs if r["mode"] == "fixed22"]) >= len(REF2_SEEDS)
        print(f"     fixed orders {list(LAYER_ORDERS)} ({len(fx)} runs, {fx[0]['params']:,} params) {fmt(L)} median {statistics.median(L):.3f}"
              + (f"  -> {decide_layers(L, ref[2])}" if done else "  (incomplete)"))
        print(f"       Mann-Whitney p (reported, not used): vs fixed 2 {mwu_p(L, ref[2]):.3f}, vs fixed 1 {mwu_p(L, ref[1]):.3f}")
    for g in gammas:
        fin = [r for r in runs if r["mode"] == "learned" and r["gamma"] == g and r["seed"] in FINAL_SEEDS]
        if len(fin) >= 5:
            rises = eo.late_rise([r["curve_eval"] for r in fin], fin[0]["steps"])
            if g in PROBE2_GAMMAS:   # follow-up: late averages against the late bar (amended 23 Oct)
                v = decide_gate_late([r["late_E"] for r in fin], [r["order"] for r in fin], ref[2])
            else:
                v = decide_gate([r["extrap"] for r in fin], [r["order"] for r in fin])
            v = "NOT CONVERGED" if rises == rises and rises > eo.LATE_RISE else v
            ok, hits = mechanism_ok([r["order_by_kind"] for r in fin])
            print(f"\n  DECISION (gamma {g}, {len(fin)} runs): {v}   median E {statistics.median(r['extrap'] for r in fin):.3f}, "
                  f"median order {statistics.median(r['order'] for r in fin):.3f}, late rise {rises:+.3f}")
            print(f"  mechanism: operation > description in {hits} of {len(fin)} runs -> {'yes' if ok else 'no'}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("probe", "final", "probe2", "layers21"))
    ap.add_argument("--gamma", type=float, default=None, help="final stage: the gamma chosen on dev")
    ap.add_argument("--steps", type=int, default=eo.STEPS)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--boxes_dir", default=os.path.join(REPO, "boxes_data"))
    ap.add_argument("--out", default=os.path.join(REPO, "results", "boxes_gate.json"))
    ap.add_argument("--save_dir", default=os.path.join(REPO, "checkpoints", "boxes_gate"),
                    help="every trained model is saved here, for timing on the faster path later")
    ap.add_argument("--device", default=bl.DEVICE)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest("cpu") else 1)
    if args.summary:
        summary(load(args.out)); return
    if args.stage is None or (args.stage == "final" and args.gamma is None):
        raise SystemExit("give --stage probe, or --stage final --gamma <gamma chosen on dev>")

    base = bl.make_args(models="delta", steps=args.steps, batch_size=args.batch_size, boxes_dir=args.boxes_dir,
                        out=args.out, device=args.device, quick_n=eo.QUICK_N)
    task = bl.BoxesTask(base.boxes_dir, None, base.max_train_ops, base.train_n, base.eval_per_bucket)
    dtask = dev_task(task, base.boxes_dir)
    print(f"    {task.train_len_desc}; dev split {[len(v) for v in dtask.eval_sets.values()]}", flush=True)
    d_ff = {m: matched_d_ff(task.vocab_size, base, m) for m in ("rule", "learned", "fixed21", "fixed22")}
    for m, f in d_ff.items():
        print(f"    {m}: d_ff {f}, params {bl.n_params(build_model(task.vocab_size, base, m, f)):,} (target {TARGET_PARAMS:,})", flush=True)

    if args.stage == "probe":
        plan = []
        for s in RULE_SEEDS:
            plan.append(("rule", 0.0, s))
            if s in PROBE_SEEDS:
                plan += [("learned", g, s) for g in PROBE_GAMMAS]
    elif args.stage == "probe2":
        plan = [("learned", g, s) for s in PROBE2_SEEDS for g in PROBE2_GAMMAS]
    elif args.stage == "layers21":
        plan = []
        for s in LAYER_SEEDS:
            plan.append(("fixed21", 0.0, s))
            if s in REF2_SEEDS:
                plan.append(("fixed22", 0.0, s))
    else:
        plan = [("learned", args.gamma, s) for s in FINAL_SEEDS]

    res = load(args.out)
    for mode, g, s in plan:
        k = key(mode, g, s)
        sig = json.dumps({"steps": args.steps, "bs": args.batch_size, "d_ff": d_ff[mode]}, sort_keys=True)
        if res["runs"].get(k, {}).get("sig") == sig:
            continue
        print(f"\n>>> {k}  ({time.strftime('%H:%M')})", flush=True)
        r = run(base, task, dtask, mode, g, s, d_ff[mode], args.device, args.steps, args.batch_size,
                log=lambda *m: print(*m, flush=True),
                save_path=os.path.join(args.save_dir, f"{mode}_g{g}_seed{s}.pt") if args.save_dir else None)
        r["sig"] = sig
        res["runs"][k] = r
        bl.save_results(args.out, res)
        print(f"<<< {k}  test E {r['extrap']:.3f}  dev E {r['dev_extrap']:.3f}  order {r['order']:.3f}  "
              + "  ".join(f"{kk} {v:.2f}" for kk, v in r["order_by_kind"].items()) + f"  ({r['minutes']:.1f} min)", flush=True)
    summary(res)


if __name__ == "__main__":
    main()
