"""Boxes order sweep: does the number of Householder factors per token (n_h)
matter on real text?

Task: Boxes (Kim & Schuster 2023), natural-language entity tracking. Train on
descriptions with at most 2 operations, evaluate exact match of the answer on
0..8 operations (3 or more = extrapolation). Model, data, training settings and
code are Shashwat's (work/boxes_lib.py, a verbatim copy of his notebook); the
only changes made here are which orders are run and how the model size is held
fixed.

Design
  * 16 000 steps (Shashwat used 8 000; amended 13 Oct before the sweep, after
    the smoke run showed higher orders learning more slowly, so a short
    budget could penalise them for speed rather than capability).
    Quick evaluation (500 examples per bucket) every 2 000 steps.
  * DeltaProduct only, n_h in {1, 2, 3}, 5 runs each (seeds 0-4), runs
    interleaved by seed so a stopped sweep is still balanced.
  * Same parameter count at every order: the MLP width is chosen so the model
    matches Shashwat's n_h = 2 Boxes model (d_ff 455, 469,120 parameters).
    More factors therefore never means more parameters.
  * Each order is stored under its own key (the notebook's key has no n_h and
    would overwrite one order with the next).

Decision rule (pre-registered 13 October 2026, before any run; also in PLAN.md)
  E(run)  = unweighted mean exact match over the buckets ops=3, ops=4, ops>=5.
  m1      = median E over the n_h = 1 runs.
  For k in {2, 3}: gain_k = median E(n_h = k) - m1,
                   above_k = number of n_h = k runs with E > m1.
  ORDER MATTERS  some k has gain_k >= 0.05 and above_k >= 80% of its runs
                 -> the learned gate goes on Boxes next.
  NO GAIN        every k has gain_k < 0.03
                 -> one factor is enough here; chess board state next.
  UNINFORMATIVE  no run of any order reaches E = 0.10 -> fix the setup, decide nothing.
  NOT CONVERGED  (checked first, after UNINFORMATIVE) for some order the median
                 quick-eval E rose by more than 0.03 over the last quarter of
                 training -> no decision; rerun at twice the steps.
  INCONCLUSIVE   anything else -> 5 more runs per order (seeds 5-9), once, and
                 the same rule on all 10; if still INCONCLUSIVE it counts as
                 NO GAIN.

Usage
  python work/exp_boxes_order.py --selftest
  python work/exp_boxes_order.py --steps 500 --seeds 0 --out results/boxes_order_smoke.json   # smoke
  nohup python -u work/exp_boxes_order.py > results/boxes_order.log 2>&1 &                   # the sweep
  python work/exp_boxes_order.py --summary                                                   # re-read results
"""

import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import boxes_lib as bl  # noqa: E402

REF_N_H, REF_D_FF, REF_PARAMS = 2, 455, 469_120  # Shashwat's Boxes delta model (results/boxes_hybrid.json)
EXTRAP = ("ops=3", "ops=4", "ops>=5")
GAIN_YES, GAIN_NO, FRAC_ABOVE = 0.05, 0.03, 0.8
LATE_RISE = 0.03  # convergence: max rise of median quick-eval E over the last quarter of training
STEPS, QUICK_N = 16000, 500


# ----------------------------------------------------------------------------- pieces with known answers

def extrap_score(final):
    """E(run): unweighted mean exact match over the extrapolation buckets."""
    return sum(final[c] for c in EXTRAP) / len(EXTRAP)


def run_key(n_h, seed):
    return f"boxes|delta|nh{n_h}|seed{seed}"


def curve_E(curve):
    """curve_eval of one run: [(step, {bucket: acc}), ...] -> {step: E}."""
    return {int(st): extrap_score(acc) for st, acc in curve}


def late_rise(curves, steps):
    """Rise of the median (over runs) quick-eval E from the last evaluation at or before 3/4 of
    training to the last evaluation. curves: list of curve_eval lists, one per run."""
    Es = [curve_E(c) for c in curves]
    common = sorted(set.intersection(*(set(e) for e in Es)))
    early = [st for st in common if st <= 0.75 * steps]
    if not early or common[-1] == early[-1]:
        return float("nan")
    return statistics.median(e[common[-1]] for e in Es) - statistics.median(e[early[-1]] for e in Es)


def decide(scores):
    """scores: {n_h: [E per run]}, must contain 1 and at least one higher order.
    -> (verdict, m1, {k: {gain, above, n}})"""
    m1 = statistics.median(scores[1])
    per = {}
    for k in sorted(scores):
        if k == 1:
            continue
        xs = scores[k]
        per[k] = {"gain": statistics.median(xs) - m1, "above": sum(x > m1 for x in xs), "n": len(xs)}
    if max(max(xs) for xs in scores.values()) < 0.10:
        v = "UNINFORMATIVE"  # no order learned to extrapolate at all: the sweep says nothing about order
    elif any(p["gain"] >= GAIN_YES and p["above"] >= FRAC_ABOVE * p["n"] for p in per.values()):
        v = "ORDER MATTERS"
    elif all(p["gain"] < GAIN_NO for p in per.values()):
        v = "NO GAIN"
    else:
        v = "INCONCLUSIVE"
    return v, m1, per


def verdict_with_convergence(v, rises):
    """rises: {n_h: late_rise}. A still-rising order means the comparison is not yet fair."""
    if v != "UNINFORMATIVE" and any(r > LATE_RISE for r in rises.values() if r == r):
        return "NOT CONVERGED"
    return v


def matched_d_ff(vocab, a, target):
    return bl.match_d_ff(vocab, a, "delta", None, target)


def reference_target(vocab, a):
    """Parameter count of the reference model (n_h = 2, d_ff 455) at this vocabulary."""
    ref = argparse.Namespace(**{**vars(a), "n_h": REF_N_H})
    return bl.n_params(bl.build(vocab, ref, "delta", None, REF_D_FF))


# ----------------------------------------------------------------------------- self-test

def selftest(device):
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}", flush=True)

    print("W1  Shashwat's notebook self-test (unchanged code)")
    check("boxes_lib.selftest passes", bl.selftest(device))

    print("W2  parameter matching")
    a = bl.make_args(n_h=REF_N_H)
    vocab = 123  # the Boxes vocabulary of the reference run
    tgt = reference_target(vocab, a)
    check("reference rebuilds to 469,120 parameters", tgt == REF_PARAMS, f"{tgt:,}")
    for n_h in (1, 2, 3):
        an = bl.make_args(n_h=n_h)
        f = matched_d_ff(vocab, an, tgt)
        p = bl.n_params(bl.build(vocab, an, "delta", None, f))
        check(f"n_h={n_h}: d_ff {f} gives {p:,} parameters, within 0.2% of target", abs(p - tgt) / tgt < 0.002)
        if n_h == 2:
            check("n_h=2 recovers the reference width 455", f == REF_D_FF, str(f))
    for n_h in (1, 3):  # control: without matching, the size would differ, so the check above can fail
        an = bl.make_args(n_h=n_h)
        p = bl.n_params(bl.build(vocab, an, "delta", None, REF_D_FF))
        check(f"control: n_h={n_h} at the fixed width 455 is off target by > 5%", abs(p - tgt) / tgt > 0.05, f"{p:,}")

    print("W3  storage keys")
    keys = {run_key(n, s) for n in (1, 2, 3) for s in range(5)}
    check("15 runs -> 15 distinct keys", len(keys) == 15)
    nb_keys = {f"boxes|delta|seed{s}" for n in (1, 2, 3) for s in range(5)}  # the notebook's scheme
    check("control: the notebook's key would collide (5 keys for 15 runs)", len(nb_keys) == 5)

    print("W4  extrapolation score")
    names = [f"ops={b}" if b < bl.BOXES_MAX_BUCKET else f"ops>={b}" for b in range(6)]
    check("EXTRAP names are bucket names the task produces", all(c in names for c in EXTRAP), str(names))
    fin = {"ops=0": 1.0, "ops=1": 1.0, "ops=2": 1.0, "ops=3": 0.9, "ops=4": 0.6, "ops>=5": 0.3}
    check("E = mean of ops 3, 4, >=5 only", abs(extrap_score(fin) - 0.6) < 1e-12, f"{extrap_score(fin)}")

    print("W5  decision rule on known answers")
    base = [0.70, 0.72, 0.74, 0.76, 0.78]  # median 0.74
    cases = [
        ("clear gain at n_h=3", {1: base, 2: base, 3: [x + 0.08 for x in base]}, "ORDER MATTERS"),
        ("no gain", {1: base, 2: [x + 0.01 for x in base], 3: [x - 0.02 for x in base]}, "NO GAIN"),
        ("higher order worse", {1: base, 2: [x - 0.1 for x in base], 3: [x - 0.1 for x in base]}, "NO GAIN"),
        ("gain 0.04: between thresholds", {1: base, 2: [x + 0.04 for x in base], 3: base}, "INCONCLUSIVE"),
        # median gain 0.06, but only 3 of 5 runs above m1: the count decides
        ("median up, only 3/5 above", {1: base, 2: [0.60, 0.65, 0.80, 0.81, 0.82]}, "INCONCLUSIVE"),
        ("nothing learned", {1: [0.0] * 5, 2: [0.01] * 5, 3: [0.05] * 5}, "UNINFORMATIVE"),
    ]
    for name, sc, want in cases:
        got = decide(sc)[0]
        check(f"{name} -> {want}", got == want, got)
    sc = cases[4][1]
    gain_only = "ORDER MATTERS" if statistics.median(sc[2]) - statistics.median(sc[1]) >= GAIN_YES else "?"
    check("control: the gain alone would have said ORDER MATTERS on that case", gain_only == "ORDER MATTERS")

    print("W5b convergence check")
    def mk(vals):  # E-values at steps 2000..8000 -> curve_eval with all buckets equal to that value
        return [(2000 * (i + 1), {c: v for c in ("ops=0", "ops=1", "ops=2") + EXTRAP}) for i, v in enumerate(vals)]
    flat = [mk([0.5, 0.7, 0.80, 0.80]) for _ in range(5)]
    rising = [mk([0.3, 0.5, 0.60, 0.70]) for _ in range(5)]
    r_flat, r_rise = late_rise(flat, 8000), late_rise(rising, 8000)
    check("plateaued curves: late rise 0", abs(r_flat) < 1e-12, f"{r_flat:+.3f}")
    check("still-rising curves: late rise 0.10 (from step 6000 to 8000)", abs(r_rise - 0.10) < 1e-9, f"{r_rise:+.3f}")
    check("control: rise measured from the start would also flag the plateaued run (so the window matters)",
          statistics.median(curve_E(c)[8000] - curve_E(c)[2000] for c in flat) > LATE_RISE)
    check("a still-rising order gives NOT CONVERGED", verdict_with_convergence("NO GAIN", {1: r_flat, 2: r_rise}) == "NOT CONVERGED")
    check("plateaued orders keep the verdict", verdict_with_convergence("NO GAIN", {1: r_flat, 2: r_flat}) == "NO GAIN")
    check("UNINFORMATIVE is not overridden", verdict_with_convergence("UNINFORMATIVE", {1: r_rise}) == "UNINFORMATIVE")

    print("W6  training step at every order")
    task = bl.RecallTask(n_keys=16, n_vals=16, train_pairs=4, eval_pairs=(4,), n_eval=8)
    for n_h in (1, 2, 3):
        an = bl.make_args(n_h=n_h, d=32, heads=2, head_dim=16, layers=1, log_every=1000, eval_every=0)
        r = bl.run_one(an, task, "delta", None, 32, 0, device, 4, 8)
        check(f"n_h={n_h}: 4 steps run, finite, result JSON-serialisable",
              not r["diverged"] and json.loads(json.dumps(r))["steps"] == 4)
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return ok


# ----------------------------------------------------------------------------- sweep

def load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"runs": {}}


def summary(res):
    runs = res["runs"]
    if not runs:
        print("no runs saved yet")
        return
    orders = sorted({r["n_h"] for r in runs.values()})
    conds = list(next(iter(runs.values()))["final"])
    print(f"\n{'=' * 78}\nBOXES ORDER SWEEP  (exact match; per-run values sorted, then median)\n{'=' * 78}")
    scores, rises = {}, {}
    for n in orders:
        rs = [r for r in runs.values() if r["n_h"] == n]
        flags = [f"seed{r['seed']}:" + ("truncated" if r["truncated"] else "") + ("DIVERGED" if r["diverged"] else "")
                 for r in rs if r["truncated"] or r["diverged"]]
        print(f"\n  n_h = {n}: {len(rs)} runs, params {rs[0]['params']:,}, d_ff {rs[0]['d_ff']}, "
              f"{statistics.median(r['minutes'] for r in rs):.1f} min/run  {' '.join(flags)}")
        for c in conds:
            vals = sorted(r["final"][c] for r in rs)
            print(f"     {c:<8} [{' '.join(f'{v:.3f}' for v in vals)}]   median {statistics.median(vals):.3f}")
        scores[n] = [extrap_score(r["final"]) for r in rs]
        print(f"     {'E':<8} [{' '.join(f'{v:.3f}' for v in sorted(scores[n]))}]   median {statistics.median(scores[n]):.3f}")
        curves = [r["curve_eval"] for r in rs if r["curve_eval"]]
        if curves:
            Es = [curve_E(c) for c in curves]
            common = sorted(set.intersection(*(set(e) for e in Es)))
            print("     quick-eval E (median over runs) by step: "
                  + "  ".join(f"{st}: {statistics.median(e[st] for e in Es):.3f}" for st in common))
            rises[n] = late_rise(curves, rs[0]["steps"])
            print(f"     rise over the last quarter: {rises[n]:+.3f}" + ("   <- STILL RISING" if rises[n] > LATE_RISE else ""))
    if 1 in scores and len(scores) > 1:
        v, m1, per = decide(scores)
        v = verdict_with_convergence(v, rises)
        n_min = min(len(x) for x in scores.values())
        print(f"\n  DECISION (pre-registered rule; E = mean of {', '.join(EXTRAP)}):  {v}"
              + ("" if n_min >= 5 else f"   [PROVISIONAL: only {n_min} run(s) for some order; the rule needs 5]"))
        print(f"     m1 (median E at n_h=1) = {m1:.3f}")
        for k, p in per.items():
            print(f"     n_h={k}: gain {p['gain']:+.3f}, {p['above']}/{p['n']} runs above m1")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", default="1,2,3")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--boxes_dir", default=os.path.join(REPO, "boxes_data"))
    ap.add_argument("--boxes_zip", default=None)
    ap.add_argument("--out", default=os.path.join(REPO, "results", "boxes_order.json"))
    ap.add_argument("--device", default=bl.DEVICE)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--summary", action="store_true", help="re-print the saved results, train nothing")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest(args.device) else 1)
    if args.summary:
        summary(load(args.out))
        return

    orders = [int(x) for x in args.orders.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    base = bl.make_args(models="delta", steps=args.steps, batch_size=args.batch_size, boxes_dir=args.boxes_dir,
                        boxes_zip=args.boxes_zip, out=args.out, device=args.device, quick_n=QUICK_N)
    task = bl.BoxesTask(base.boxes_dir, base.boxes_zip, base.max_train_ops, base.train_n, base.eval_per_bucket)
    print(f"    {task.train_len_desc}", flush=True)
    target = reference_target(task.vocab_size, base)
    if target != REF_PARAMS:
        print(f"!!  reference model has {target:,} parameters here, not {REF_PARAMS:,}: the vocabulary differs "
              f"from the reference run ({task.vocab_size}). Sizes are still matched to each other.", flush=True)
    cfg = {}
    for n in orders:
        a = argparse.Namespace(**{**vars(base), "n_h": n})
        a.d_ff = matched_d_ff(task.vocab_size, a, target)
        cfg[n] = a
        print(f"    n_h={n}: d_ff {a.d_ff}, params {bl.n_params(bl.build(task.vocab_size, a, 'delta', None, a.d_ff)):,} "
              f"(target {target:,})", flush=True)

    res = load(args.out)
    for s in seeds:
        for n in orders:
            a = cfg[n]
            key = run_key(n, s)
            sig = json.dumps({k: getattr(a, k, None) for k in bl.SIG_KEYS}, sort_keys=True)
            if res["runs"].get(key, {}).get("sig") == sig:
                continue
            print(f"\n>>> {key}  ({time.strftime('%H:%M')})", flush=True)
            r = bl.run_one(a, task, "delta", None, a.d_ff, s, a.device, a.steps, a.batch_size)
            r.update(n_h=n, sig=sig, extrap=extrap_score(r["final"]))
            res["runs"][key] = r
            bl.save_results(args.out, res)
            print(f"<<< {key}  " + "  ".join(f"{k} {x:.3f}" for k, x in r["final"].items())
                  + f"   E {r['extrap']:.3f}  ({r['minutes']:.1f} min)", flush=True)
    summary(res)


if __name__ == "__main__":
    main()
