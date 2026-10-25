"""
Fair-baseline timing on S5: is the faster path's speed-up the gate's, or the kernel's?

    python work/time_s5_kernels.py --selftest
    python -u work/time_s5_kernels.py --ckpt checkpoints/gate_B/learned_g_0_1_hard_1_g0.1_seed12.pt \
        checkpoints/gate_B/learned_g_0_1_hard_1_g0.1_seed13.pt \
        checkpoints/gate_restart/learned_g_0_1_hard_1_g0.1_seed31.pt > results/s5_kernel_timing.log 2>&1

Pre-registered in PLAN.md ("Fair-baseline timing on S5", 25 October) before any run.

WHY. On 25 Oct the faster path with fla's float32-state recurrent kernel (Rrec) ran at
0.63-0.64 of the best fixed-4-factor time -- but that baseline used only the bf16 chunk
kernel. If the recurrent kernel is simply faster at these sizes, part of the 0.63 is the
kernel's, not the gate's. Here the fixed-4 baseline is given every option the faster path
has: both kernels, with and without fused factor projections.

CONFIGURATIONS (inference, the learned gate inside the forward pass where present):
  f4_pf_chunk  f4_fu_chunk  f4_pf_rec  f4_fu_rec   fixed 4 factors: per-factor or fused
                                                   projections x chunk or recurrent kernel
  f1_fu_chunk  f1_fu_rec                           fixed 1 factor: the floor
  g_A_rec                                          the gated model, every factor computed
                                                   (fused Path A), recurrent kernel: the
                                                   gate's cost without any skipping
  g_R_chunk  g_R_rec                               the gated model, faster path (fused +
                                                   ragged), chunk or recurrent kernel
SIZES  batch {1, 8, 32} x length {512, 2048}: test sequences (128 tokens) joined end to end,
       so the gate sees the token mix it learned (timing only; accuracy is ragged_fix.json).
PROTOCOL  per size: 3 warm-up passes per configuration, then 20 rounds with the
       configurations in a seeded random order; a pass = 3 forward calls; GPU idle.

DECISION (per model and size), written before any run
  best4 = the fixed-4 configuration with the lowest median time at that size.
  ratio = time(g_R_rec) / time(best4) per round; median and bootstrap 95% interval.
  GATE SAVING       the interval lies entirely at or below 0.90 (the original rule:
                    at least 10% faster than the best fixed-4 layer)
  NO GATE SAVING    otherwise
  The claim states the sizes with GATE SAVING on all three models, and their ratios.
REPORTED EITHER WAY: the kernel's own effect on fixed 4 (best recurrent / best chunk),
the gate's cost without skipping (g_A_rec / f4_fu_rec), the floor (best fixed-1 / best4),
and g_R_chunk / best4 (the 13 Oct path against the fair baseline).

CHECKS
  K1 every configuration meant to use the recurrent kernel called it, and every chunk
     configuration did not (a call counter on the swapped kernel): a swap that silently
     failed would otherwise time the chunk kernel under the recurrent label.
  K2 the gate really skips: mean factors per token on the timed batch (reported; the
     learned allocation is ~1.30).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "work"))

from check_ragged_fix import make_kernel  # noqa: E402  (CPU-safe)

SIZES = ((1, 512), (8, 512), (32, 512), (1, 2048), (8, 2048), (32, 2048))
WARMUP, ROUNDS, REPS, BOOT, SEED = 3, 20, 3, 2000, 1234
SAVING_MAX = 0.90
F4 = ("f4_pf_chunk", "f4_fu_chunk", "f4_pf_rec", "f4_fu_rec")
CONFIGS = {  # name: (model key, fused, ragged, kernel)
    "f4_pf_chunk": ("f4", False, False, None), "f4_fu_chunk": ("f4", True, False, None),
    "f4_pf_rec": ("f4", False, False, "recurrent"), "f4_fu_rec": ("f4", True, False, "recurrent"),
    "f1_fu_chunk": ("f1", True, False, None), "f1_fu_rec": ("f1", True, False, "recurrent"),
    "g_A_rec": ("g", True, False, "recurrent"),
    "g_R_chunk": ("g", True, True, None), "g_R_rec": ("g", True, True, "recurrent"),
}


# ---------------------------------------------------------------------------
# kernel swap with a call counter
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def counted_swap(module, kind, recurrent, counter, key):
    """Swap the layer module's kernels for `kind` (None = leave them), counting calls to
    the swapped kernels under counter[key]. Always restores."""
    if kind is None:
        yield
        return
    saved = (module.chunk_gated_delta_rule, module.chunk_delta_rule)

    def counting(fn):
        def wrapped(*a, **k):
            counter[key] = counter.get(key, 0) + 1
            return fn(*a, **k)
        return wrapped

    module.chunk_gated_delta_rule = counting(make_kernel(kind, True, saved[0], recurrent))
    module.chunk_delta_rule = counting(make_kernel(kind, False, saved[1], recurrent))
    try:
        yield
    finally:
        module.chunk_gated_delta_rule, module.chunk_delta_rule = saved


# ---------------------------------------------------------------------------
# harness and statistics (the same design as work/time_boxes_orders.py)
# ---------------------------------------------------------------------------

def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def run_rounds(passes, rounds=ROUNDS, warmup=WARMUP, seed=SEED, device="cpu"):
    names = list(passes)
    for n in names:
        for _ in range(warmup):
            passes[n]()
    rng = random.Random(seed)
    times, orders = {n: [] for n in names}, []
    for _ in range(rounds):
        order = names[:]
        rng.shuffle(order)
        orders.append(order)
        for n in order:
            sync(device)
            t = time.perf_counter()
            passes[n]()
            sync(device)
            times[n].append(time.perf_counter() - t)
    return times, orders


def boot_ci(xs, n=BOOT, seed=SEED):
    rng = random.Random(seed)
    meds = sorted(statistics.median(rng.choices(xs, k=len(xs))) for _ in range(n))
    return meds[int(0.025 * (n - 1))], meds[int(0.975 * (n - 1))]


def best_of(times, names):
    """The configuration among `names` with the lowest median time."""
    return min(names, key=lambda n: statistics.median(times[n]))


def ratio_rounds(times, a, b):
    return [x / y for x, y in zip(times[a], times[b])]


def decide(ci):
    return "GATE SAVING" if ci[1] <= SAVING_MAX + 1e-12 else "NO GATE SAVING"


def joined(te_x, B, T):
    """B rows of length T made of consecutive test sequences."""
    per = te_x.shape[1]
    if T % per:
        raise ValueError(f"length {T} is not a multiple of the sequence length {per}")
    need = B * (T // per)
    if te_x.shape[0] < need:
        raise ValueError(f"need {need} test sequences, have {te_x.shape[0]}")
    return te_x[:need].reshape(B, T)


def summarise(times):
    """Per-size summary from the per-round times of every configuration."""
    b4 = best_of(times, F4)
    r = ratio_rounds(times, "g_R_rec", b4)
    ci = boot_ci(r)
    med = {n: statistics.median(v) for n, v in times.items()}
    best_chunk4 = min(med[n] for n in F4 if n.endswith("chunk"))
    best_rec4 = min(med[n] for n in F4 if n.endswith("rec"))
    return {"best4": b4, "ratio_median": statistics.median(r), "ratio_ci95": ci, "decision": decide(ci),
            "median_s": med,
            "kernel_effect_f4": best_rec4 / best_chunk4,
            "gate_cost_no_skip": med["g_A_rec"] / med["f4_fu_rec"],
            "floor_f1": min(med["f1_fu_chunk"], med["f1_fu_rec"]) / med[b4],
            "g_R_chunk_vs_best4": med["g_R_chunk"] / med[b4]}


# ---------------------------------------------------------------------------
# self-test (CPU)
# ---------------------------------------------------------------------------

def selftest() -> bool:
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""), flush=True)

    print("time_s5_kernels self-test (CPU)")
    print("T1  counted swap")
    calls = []

    def orig(**kw):
        calls.append("orig")
        return kw["q"], None

    def rec(**kw):
        calls.append("rec")
        return kw["q"], None

    class Mod:
        chunk_gated_delta_rule = staticmethod(orig)
        chunk_delta_rule = staticmethod(orig)
    m, counter = Mod(), {}
    q = torch.zeros(1, 3, 2, 4, dtype=torch.bfloat16)
    args = dict(q=q, k=q, v=q, g=torch.zeros(1, 3, 2), beta=torch.zeros(1, 3, 2, dtype=torch.bfloat16),
                use_qk_l2norm_in_kernel=True)
    with counted_swap(m, "recurrent", rec, counter, "x"):
        m.chunk_gated_delta_rule(**args)
        m.chunk_gated_delta_rule(**args)
    check("swapped kernel called and counted (2 calls, both reach the recurrent kernel)",
          counter.get("x") == 2 and calls == ["rec", "rec"], f"{counter} {calls}")
    calls.clear()
    with counted_swap(m, None, rec, counter, "y"):
        m.chunk_gated_delta_rule(**args)
    check("kind None: original kernel, nothing counted", calls == ["orig"] and "y" not in counter)
    check("restored after use", m.chunk_gated_delta_rule is Mod.chunk_gated_delta_rule)
    try:
        with counted_swap(m, "recurrent", rec, counter, "z"):
            raise ValueError
    except ValueError:
        pass
    check("restored after an exception", m.chunk_gated_delta_rule is Mod.chunk_gated_delta_rule)

    print("T2  harness and statistics on known answers (sleeps)")
    nap = lambda s: (lambda: time.sleep(s))
    passes = {"f4_pf_chunk": nap(0.010), "f4_fu_chunk": nap(0.009), "f4_pf_rec": nap(0.008), "f4_fu_rec": nap(0.006),
              "f1_fu_chunk": nap(0.004), "f1_fu_rec": nap(0.003), "g_A_rec": nap(0.007),
              "g_R_chunk": nap(0.005), "g_R_rec": nap(0.003)}
    t, orders = run_rounds(passes, rounds=10, warmup=1)
    s = summarise(t)
    check("best4 = the fastest fixed-4 configuration (f4_fu_rec), not a faster non-fixed-4 one",
          s["best4"] == "f4_fu_rec", s["best4"])
    check("ratio g_R_rec / best4 ~ 3/6 = 0.5 and GATE SAVING", 0.4 < s["ratio_median"] < 0.6 and s["decision"] == "GATE SAVING",
          f"{s['ratio_median']:.3f} {s['ratio_ci95']}")
    check("kernel effect on fixed 4 = best rec / best chunk ~ 6/9", 0.6 < s["kernel_effect_f4"] < 0.73, f"{s['kernel_effect_f4']:.3f}")
    check("gate cost without skipping = g_A_rec / f4_fu_rec ~ 7/6", 1.08 < s["gate_cost_no_skip"] < 1.26, f"{s['gate_cost_no_skip']:.3f}")
    check("rounds run in varying, reproducible order",
          len({tuple(o) for o in orders}) > 1 and run_rounds({k: nap(0) for k in passes}, rounds=10, warmup=0)[1] == orders)
    passes2 = dict(passes, g_R_rec=nap(0.0057))
    s2 = summarise(run_rounds(passes2, rounds=10, warmup=1)[0])
    check("a ratio near 0.95 is NO GATE SAVING", s2["decision"] == "NO GATE SAVING", f"{s2['ratio_median']:.3f} {s2['ratio_ci95']}")
    check("decision at the boundary: interval upper 0.90 saves, 0.9001 does not",
          decide((0.8, 0.90)) == "GATE SAVING" and decide((0.8, 0.9001)) == "NO GATE SAVING")
    check("interval of a constant is that constant", boot_ci([0.6] * 20) == (0.6, 0.6))

    print("T3  joined inputs")
    te = torch.arange(10 * 4).reshape(10, 4)
    x = joined(te, 2, 8)
    check("2 rows of length 8 from 4 consecutive 4-token sequences", x.shape == (2, 8) and x[1, 0].item() == 8)
    bad = []
    for B, T in ((3, 6), (6, 8)):
        try:
            joined(te, B, T)
            bad.append(False)
        except ValueError:
            bad.append(True)
    check("refuses a length that is not a multiple, and too few sequences", all(bad))
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return ok


# ---------------------------------------------------------------------------
# GPU run
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt", nargs="*", default=[])
    ap.add_argument("--out", default=str(ROOT / "results" / "s5_kernel_timing.json"))
    args = ap.parse_args()
    if not selftest():
        sys.exit(1)
    if args.selftest:
        return
    if not args.ckpt:
        sys.exit("give --ckpt with trained gated S5 checkpoints")

    import fla.layers.gated_deltaproduct as gdp
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule as fr
    from exp_matched_compute import build, load, DEV
    from exp_gate_train import set_gate_hard, collect_n_t
    from check_ragged import set_fused, set_ragged
    from check_ragged_trained import data_csv
    from time_boxes_orders import gpu_busy

    if gpu_busy():
        sys.exit("a training script is running: timings would be unreliable")

    def make_pass(model, x, fused, ragged, kind, counter, key, adaptive):
        # set_ragged only exists for adaptive layers; a fixed model is never ragged
        def f():
            set_fused(model, fused)
            if adaptive:
                set_ragged(model, ragged)
            with counted_swap(gdp, kind, fr, counter, key), torch.inference_mode():
                for _ in range(REPS):
                    model(input_ids=x, use_cache=False)
            set_fused(model, False)
            if adaptive:
                set_ragged(model, False)
        return f

    res = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "rounds": ROUNDS, "reps": REPS,
           "models": []}
    all_k1 = True
    for path in args.ckpt:
        ck = torch.load(path, map_location=DEV, weights_only=True)
        b, d = ck["build"], ck["data"]
        g = build(**b)
        g.load_state_dict(ck["state_dict"])
        g.eval()
        set_gate_hard(g, True)
        common = dict(n_heads=b["n_heads"], head_dim=b["head_dim"], hidden=b["hidden"], n_layers=b["n_layers"],
                      vocab=b["vocab"], seed=0)
        models = {"g": g, "f4": build(n_h=4, adaptive=False, **common).eval(),
                  "f1": build(n_h=1, adaptive=False, **common).eval()}
        _, _, te_x, _ = load(data_csv(d), d["n_train"], d["n_test"])
        name = Path(path).name
        print("=" * 78 + f"\n{name}", flush=True)
        entry = {"checkpoint": name, "sizes": []}
        for B, T in SIZES:
            x = joined(te_x, B, T)
            with torch.inference_mode():
                g(input_ids=x, use_cache=False)
            order = torch.stack(collect_n_t(g)).float().mean().item()
            counter = {}
            passes = {c: make_pass(models[mk], x, fu, ra, kd, counter, c, mk == "g") for c, (mk, fu, ra, kd) in CONFIGS.items()}
            times, _ = run_rounds(passes, device=DEV)
            k1 = all((counter.get(c, 0) > 0) == (CONFIGS[c][3] is not None) for c in CONFIGS)
            all_k1 &= k1
            s = summarise(times)
            s.update(batch=B, length=T, gate_order=order, k1=k1, calls=counter)
            entry["sizes"].append(s)
            ms = lambda n: 1000 * s["median_s"][n] / REPS
            print(f"  {B:>2} x {T:<5} K1 {'PASS' if k1 else 'FAIL'}  order {order:.3f}  best4 {s['best4']} {ms(s['best4']):.2f} ms"
                  f" | g_R_rec {ms('g_R_rec'):.2f} ms -> ratio {s['ratio_median']:.3f} "
                  f"({s['ratio_ci95'][0]:.3f}-{s['ratio_ci95'][1]:.3f}) {s['decision']}", flush=True)
            print(f"           fixed-4 rec/chunk {s['kernel_effect_f4']:.3f}  gate cost no-skip {s['gate_cost_no_skip']:.3f}"
                  f"  floor f1/best4 {s['floor_f1']:.3f}  g_R_chunk/best4 {s['g_R_chunk_vs_best4']:.3f}  | "
                  + "  ".join(f"{n} {ms(n):.2f}" for n in CONFIGS), flush=True)
        res["models"].append(entry)

    print("\n" + "=" * 78)
    if not all_k1:
        print("K1 FAILED somewhere: a kernel swap did not behave as labelled -- no claim from this run")
    claim = []
    for i, (B, T) in enumerate(SIZES):
        ds = [m["sizes"][i]["decision"] for m in res["models"]]
        rs = [m["sizes"][i]["ratio_median"] for m in res["models"]]
        line = f"  {B:>2} x {T:<5} " + "  ".join(f"{r:.3f}" for r in rs) + "  -> " + (
            "GATE SAVING on every model" if all(d == "GATE SAVING" for d in ds) else
            f"GATE SAVING on {sum(d == 'GATE SAVING' for d in ds)} of {len(ds)}")
        print(line)
        if all(d == "GATE SAVING" for d in ds) and all_k1:
            claim.append({"batch": B, "length": T, "ratios": rs})
    res["claim_sizes"] = claim
    res["all_k1"] = all_k1
    print(f"CLAIM: GATE SAVING on every model at {len(claim)} of {len(SIZES)} sizes")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
