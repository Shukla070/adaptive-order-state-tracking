"""
Ragged expansion ("Path B") on the GPU: is it exact, and is it faster?

    python work/check_ragged.py --selftest     # correctness only, ~1 min
    python work/check_ragged.py                # correctness, then the timing grid
    python work/check_ragged.py --out results/ragged_bench.json

PART 1 — CORRECTNESS (the real Triton kernel, bf16)
  A 1-layer model at the experiment size (12 heads x 32, hidden 128, K = 4) is
  run twice on the same input and the same hard gates: once with Path A (every
  factor computed, closed ones multiplied by zero) and once with ragged
  expansion (closed factors dropped, sequences packed with cu_seqlens). The
  layer outputs must agree to within bf16 noise, and four deliberately broken ragged
  variants must NOT agree. The float64 proof of the indexing is
  `python src/ragged_expansion.py`; this checks the real kernel.

  Pass criteria (fixed before the first run):
    C1  ragged vs Path A, random orders 0-4, compared on the layer's output:
        relative L2 difference ||a - b|| / ||a|| <= 1e-2 (bf16 precision is
        ~8e-3 per element)
    C1b the float64 reference reproduces the kernel's Path A recurrence output
        (relative L2 <= 5e-2), so it can be used as the judge in C1c
    C1c added 11 Oct, after a run of the earlier version failed only on an
        argmax-agreement criterion (0.9907 vs 0.999) on a random-weight model
        whose logits are near-ties. Rather than loosen anything, this asks the
        sharper question directly: on the recurrence output, ragged's error
        against the float64 reference must be no larger than 2x Path A's own
        error against it (+1e-4). bf16 noise passes; a real bug cannot.
    C2  the same with every gate open
    C3  each mutant: relative L2 difference > 0.1 (clearly detected)
    C4  harness: timing a function that does 2x the work reads 1.6x-2.5x
    C5-C8 (added 12 Oct, before their first run) the fused-factor path:
        C5  fused Path A vs per-factor Path A: relative L2 <= 1e-2
        C6  fused ragged vs per-factor Path A: relative L2 <= 1e-2
        C6b fused ragged vs float64 reference <= 2x per-factor Path A's error
        C7  three mutants on the fused ragged path: relative L2 > 0.1
        C8  a fixed (non-adaptive) n_h = 4 layer, fused vs per-factor: <= 1e-2

PART 2 — TIMING (forward pass only, inference mode, cuda events)
  For each (batch, length) — default batches 1, 8, 32 and lengths 128, 512,
  2048, kept under the 6 GB card's limit — fixed n_h = 1, 2, 3, 4 (layers built with that many
  factors), the adaptive layer with Path A, and the adaptive layer with ragged
  expansion, under two allocations:
    s5mix   order 4 with probability 0.1, else 1 — the allocation the learned
            gate reached in 13 of 20 runs on the S5 mix (mean 1.30)
    all4    every token at order 4 — the worst case, measures packing overhead
  Median of 30 timed repeats after 5 warm-up calls. Model weights are random:
  speed does not depend on them, and correctness is Part 1.

  Reading, fixed before the first run: a speedup is claimed only where ragged
  s5mix is at least 10% faster than fixed n_h = 4 at the same (batch, length),
  with Part 1 passing. Every cell of the grid is reported either way,
  including the ones where it is slower.
  Revised 12 Oct, before the fused path's first run (see timing()): the claim
  now uses fused ragged + the gate's own cost, against the faster of the
  per-factor and fused 4-factor baselines.

Nothing here trains. Inference only: ragged expansion cannot train a gate
(closed factors receive no gradient).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "work"))

from exp_matched_compute import build, DEV  # noqa: E402
from oracle_order import set_external_gates  # noqa: E402
from ragged_expansion import path_a_core, reference_gated_delta  # noqa: E402

K = 4
VOCAB = 128


def adaptive_layers(model):
    layers = [m for m in model.modules()
              if getattr(m, "adaptive_order", False) and hasattr(m, "external_gates")]
    if not layers:
        raise RuntimeError("no adaptive-order layer found")
    return layers


def dp_layers(model):
    """Every DeltaProduct layer, adaptive or not."""
    layers = [m for m in model.modules() if hasattr(m, "k_projs") and hasattr(m, "num_householder")]
    if not layers:
        raise RuntimeError("no DeltaProduct layer found")
    return layers


def set_fused(model, on: bool):
    for m in dp_layers(model):
        m.fused_factors = on


def set_ragged(model, on: bool, mutant: str = ""):
    for m in adaptive_layers(model):
        m.ragged = on
        m.ragged_mutant = mutant


def prefix_gates(n_t: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
    ar = torch.arange(K, device=n_t.device).view(1, 1, K)
    return (ar < n_t.unsqueeze(-1)).to(dtype)


def orders(kind: str, B: int, T: int, gen: torch.Generator) -> torch.Tensor:
    u = torch.rand(B, T, generator=gen, device=DEV)
    if kind == "s5mix":
        return torch.where(u < 0.1, 4, 1).long()
    if kind == "all4":
        return torch.full((B, T), 4, device=DEV, dtype=torch.long)
    if kind == "random":   # orders 0..4, used for correctness
        return torch.randint(0, K + 1, (B, T), generator=gen, device=DEV)
    raise ValueError(kind)


@torch.inference_mode()
def logits(model, x):
    """Output of the adaptive layer itself (not the model's logits: in a
    1-layer model the residual path can hide a broken layer in the logits)."""
    store = []
    hooks = [m.register_forward_hook(lambda mod, inp, out: store.append(out[0].float()))
             for m in dp_layers(model)]
    try:
        model(input_ids=x, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return torch.cat([o.reshape(-1) for o in store])


def compare(a, b):
    """(relative L2 difference, relative max difference)."""
    l2 = ((a - b).norm() / a.norm().clamp_min(1e-12)).item()
    mx = ((a - b).abs().max() / a.abs().max().clamp_min(1e-12)).item()
    return l2, mx


def capture(model, x, ragged_on: bool) -> dict:
    """Run one forward and return the adaptive layer's recurrence inputs and
    output (o, shape (B, T, H, Dv)), via the layer's `capture` hook."""
    layer = adaptive_layers(model)[0]
    layer.capture = []
    set_ragged(model, ragged_on)
    try:
        with torch.inference_mode():
            model(input_ids=x, use_cache=False)
        c = layer.capture[0]
    finally:
        layer.capture = None
        set_ragged(model, False)
    if "o" not in c:
        raise RuntimeError("layer did not record its output; is the capture patch applied?")
    return c


def reference_output(c: dict) -> torch.Tensor:
    """Path A's recurrence recomputed in float64 with the sequential reference
    kernel, from the captured inputs. Independent of the ragged code."""
    H = c["num_heads"]

    def heads(t):
        B, T, _ = t.shape
        return t.double().reshape(B, T, H, -1)

    q = heads(c["q"])
    ks = [heads(k) for k in c["ks"]]
    vs = [heads(v) for v in c["vs"]]
    betas = [b.double() for b in c["betas"]]
    g = None if c["g"] is None else c["g"].double()
    return path_a_core(q, ks, vs, betas, g,
                       lambda q_, k_, v_, g_, b_: reference_gated_delta(q_, k_, v_, g_, b_, None))


def rel_err(a, ref) -> float:
    return ((a.double() - ref).norm() / ref.norm().clamp_min(1e-12)).item()


# ---------------------------------------------------------------------------
# Part 1
# ---------------------------------------------------------------------------

def correctness() -> bool:
    ok_all = True

    def check(name, cond, detail=""):
        nonlocal ok_all
        ok_all &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}   {detail}")

    print("PART 1 — correctness, real kernel, bf16")
    model = build(n_h=K, adaptive=True, n_heads=12, head_dim=32, hidden=128,
                  n_layers=1, vocab=VOCAB, seed=0).eval()
    gen = torch.Generator(device=DEV).manual_seed(0)
    B, T = 8, 256
    x = torch.randint(0, VOCAB, (B, T), generator=gen, device=DEV)

    # C0: Path A against itself — the noise floor the tolerance is set above
    n_t = orders("random", B, T, gen)
    set_external_gates(model, prefix_gates(n_t))
    set_ragged(model, False)
    a1, a2 = logits(model, x), logits(model, x)
    rel0, mx0 = compare(a1, a2)
    print(f"  [info] Path A run-to-run: relative L2 {rel0:.2e}, max {mx0:.2e}")

    # C1: random orders 0..4
    set_ragged(model, True)
    r = logits(model, x)
    rel, mx = compare(a1, r)
    check("C1 ragged == Path A, random orders 0-4", rel <= 1e-2,
          f"relative L2 {rel:.2e} (max {mx:.2e}), "
          f"{int(n_t.clamp(min=1).sum())} micro-steps instead of {K * B * T}")

    # C1b / C1c: judge both paths against a float64 reference of the recurrence
    ca = capture(model, x, ragged_on=False)
    cb = capture(model, x, ragged_on=True)
    ref = reference_output(ca)
    err_a = rel_err(ca["o"], ref)
    err_b = rel_err(cb["o"], ref)
    check("C1b float64 reference reproduces the kernel's Path A", err_a <= 5e-2,
          f"Path A vs reference: relative L2 {err_a:.2e}")
    check("C1c ragged is as close to the reference as Path A is",
          err_b <= 2 * err_a + 1e-4,
          f"ragged vs reference {err_b:.2e}, Path A vs reference {err_a:.2e}")

    # C2: all open
    n4 = orders("all4", B, T, gen)
    set_external_gates(model, prefix_gates(n4))
    set_ragged(model, False)
    a = logits(model, x)
    set_ragged(model, True)
    r = logits(model, x)
    rel, mx = compare(a, r)
    check("C2 all gates open: ragged == Path A", rel <= 1e-2,
          f"relative L2 {rel:.2e} (max {mx:.2e})")

    # C3: mutants must be detected
    set_external_gates(model, prefix_gates(n_t))
    set_ragged(model, False)
    a = logits(model, x)
    for mut in ("no_cu_seqlens", "query_at_first", "read_at_first", "decay_at_last"):
        set_ragged(model, True, mutant=mut)
        r = logits(model, x)
        rel, mx = compare(a, r)
        if mut == "decay_at_last" and not any(
                getattr(m, "use_forget_gate", False) for m in adaptive_layers(model)):
            print(f"  [skip] C3 {mut}: this model has no decay term (use_forget_gate=False)")
            continue
        check(f"C3 mutant '{mut}' detected", rel > 0.1,
              f"relative L2 {rel:.2e} (max {mx:.2e})")
    set_ragged(model, False)
    set_external_gates(model, None)

    # C5-C8: the fused-factor path (one matmul / one conv for all K factors)
    set_external_gates(model, prefix_gates(n_t))
    set_fused(model, False)
    a_unf = logits(model, x)                          # per-factor Path A
    ca = capture(model, x, ragged_on=False)
    err_a = rel_err(ca["o"], reference_output(ca))
    set_fused(model, True)
    a_f = logits(model, x)
    rel, mx = compare(a_unf, a_f)
    check("C5 fused Path A == per-factor Path A", rel <= 1e-2,
          f"relative L2 {rel:.2e} (max {mx:.2e})")
    set_ragged(model, True)
    r_f = logits(model, x)
    rel, mx = compare(a_unf, r_f)
    check("C6 fused ragged == per-factor Path A", rel <= 1e-2,
          f"relative L2 {rel:.2e} (max {mx:.2e})")
    cf = capture(model, x, ragged_on=True)            # fused + ragged
    err_f = rel_err(cf["o"], reference_output(cf))
    check("C6b fused ragged is as close to the float64 reference as Path A is",
          err_f <= 2 * err_a + 1e-4,
          f"fused ragged vs reference {err_f:.2e}, per-factor Path A {err_a:.2e}")
    for mut in ("no_cu_seqlens", "query_at_first", "read_at_first"):
        set_ragged(model, True, mutant=mut)
        rel, mx = compare(a_unf, logits(model, x))
        check(f"C7 fused mutant '{mut}' detected", rel > 0.1,
              f"relative L2 {rel:.2e} (max {mx:.2e})")
    set_ragged(model, False)
    set_fused(model, False)
    set_external_gates(model, None)
    fixed = build(n_h=K, adaptive=False, n_heads=12, head_dim=32, hidden=128,
                  n_layers=1, vocab=VOCAB, seed=0).eval()
    f_unf = logits(fixed, x)
    set_fused(fixed, True)
    f_f = logits(fixed, x)
    rel, mx = compare(f_unf, f_f)
    check("C8 fixed n_h=4 layer: fused == per-factor", rel <= 1e-2,
          f"relative L2 {rel:.2e} (max {mx:.2e})")
    del fixed

    # C4: the timing harness can see a 2x difference
    m = torch.randn(1024, 1024, device=DEV, dtype=torch.bfloat16)

    def one():
        return m @ m

    def two():
        return (m @ m) @ m

    t1, t2 = time_fn(one), time_fn(two)
    check("C4 timing harness sees 2x work", 1.6 <= t2 / t1 <= 2.5,
          f"{t1:.3f} ms vs {t2:.3f} ms, ratio {t2 / t1:.2f}")
    print("PART 1: ALL PASS" if ok_all else "PART 1: SOME CHECKS FAILED")
    return ok_all


# ---------------------------------------------------------------------------
# Part 2
# ---------------------------------------------------------------------------

def time_fn(fn, warmup=5, reps=30) -> float:
    """Median milliseconds per call, measured with cuda events."""
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def timing(grid_b, grid_t, out_path):
    """Forward-pass time of every variant, per (batch, length).

    Columns (ms):
      fixK      fixed n_h = K, upstream's per-factor code
      fixKf     fixed n_h = K, fused factors (one matmul / conv for all K)
      A         adaptive, per-factor, all factors computed (Path A)
      B         adaptive, per-factor, ragged expansion (version 2 packing)
      Bf        adaptive, fused factors, ragged expansion
      Bf_all4   the same with every factor kept (overhead check)
      gate      the learned halting gate alone (an MLP per token); the
                timed adaptive runs use external gates, so its cost is
                added separately for the claim
    The claim (fixed before the run, 12 Oct) compares Bf + gate with the
    FASTER of the two 4-factor baselines (fix4, fix4f), so a saving that only
    comes from fusing projections, which a fixed layer can also do, is not
    counted as adaptivity.
    """
    print("\nPART 2 — forward-pass time, ms (median of 30), inference mode")
    fixed = {n: build(n_h=n, adaptive=False, n_heads=12, head_dim=32, hidden=128,
                      n_layers=1, vocab=VOCAB, seed=0).eval() for n in (1, 2, 3, 4)}
    adapt = build(n_h=K, adaptive=True, n_heads=12, head_dim=32, hidden=128,
                  n_layers=1, vocab=VOCAB, seed=0).eval()
    gate = adaptive_layers(adapt)[0].halting_gate
    gen = torch.Generator(device=DEV).manual_seed(1)
    rows = []
    hdr = (f"{'batch':>5} {'len':>5} | {'fix1f':>6} {'fix2f':>6} {'fix3f':>6} {'fix4f':>6} "
           f"{'fix4':>6} | {'A':>6} {'B':>6} {'Bf':>6} {'Bf_all4':>7} {'gate':>5} | "
           f"{'(Bf+gate)/best4':>15}")
    print(hdr)
    print("-" * len(hdr))

    def fwd(mdl, x):
        return lambda: mdl(input_ids=x, use_cache=False)

    for B in grid_b:
        for T in grid_t:
            try:
                torch.cuda.reset_peak_memory_stats()
                x = torch.randint(0, VOCAB, (B, T), generator=gen, device=DEV)
                row = dict(batch=B, length=T)
                for n, mdl in fixed.items():
                    set_fused(mdl, False)
                    row[f"fixed{n}_ms"] = time_fn(fwd(mdl, x))
                    set_fused(mdl, True)
                    row[f"fixed{n}_fused_ms"] = time_fn(fwd(mdl, x))
                    set_fused(mdl, False)
                n_mix = orders("s5mix", B, T, gen)
                row["s5mix_mean_order"] = n_mix.float().mean().item()
                set_external_gates(adapt, prefix_gates(n_mix))
                set_fused(adapt, False)
                set_ragged(adapt, False)
                row["pathA_s5mix_ms"] = time_fn(fwd(adapt, x))
                set_ragged(adapt, True)
                row["ragged_s5mix_ms"] = time_fn(fwd(adapt, x))
                set_fused(adapt, True)
                row["ragged_fused_s5mix_ms"] = time_fn(fwd(adapt, x))
                set_external_gates(adapt, prefix_gates(orders("all4", B, T, gen)))
                row["ragged_fused_all4_ms"] = time_fn(fwd(adapt, x))
                set_ragged(adapt, False)
                set_fused(adapt, False)
                h = torch.randn(B, T, 128, device=DEV, dtype=torch.bfloat16)
                row["gate_ms"] = time_fn(lambda: gate(h, hard=True))
                row["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
                if row["peak_mem_gb"] > 5.0:
                    # WSL spills past VRAM into system RAM without failing,
                    # which would make every time in this row meaningless.
                    print(f"  WARNING: peak memory {row['peak_mem_gb']:.1f} GB at "
                          f"batch {B}, length {T}: may have spilled; distrust this row")
                row["best4_ms"] = min(row["fixed4_ms"], row["fixed4_fused_ms"])
                row["claim_ratio"] = (row["ragged_fused_s5mix_ms"] + row["gate_ms"]) / row["best4_ms"]
                row["ragged_vs_fixed4"] = row["ragged_s5mix_ms"] / row["fixed4_ms"]
                rows.append(row)
                print(f"{B:>5} {T:>5} | {row['fixed1_fused_ms']:6.2f} {row['fixed2_fused_ms']:6.2f} "
                      f"{row['fixed3_fused_ms']:6.2f} {row['fixed4_fused_ms']:6.2f} "
                      f"{row['fixed4_ms']:6.2f} | {row['pathA_s5mix_ms']:6.2f} "
                      f"{row['ragged_s5mix_ms']:6.2f} {row['ragged_fused_s5mix_ms']:6.2f} "
                      f"{row['ragged_fused_all4_ms']:7.2f} {row['gate_ms']:5.2f} | "
                      f"{row['claim_ratio']:15.2f}")
            except torch.cuda.OutOfMemoryError:
                print(f"{B:>5} {T:>5} | out of memory, skipped")
                torch.cuda.empty_cache()
            finally:
                set_external_gates(adapt, None)
                set_ragged(adapt, False)
                set_fused(adapt, False)
    claimed = [r for r in rows if r["claim_ratio"] <= 0.90]
    print("\nReading (fixed in advance, 12 Oct): a speedup is claimed only where")
    print("fused ragged + gate is at least 10% faster than the FASTER 4-factor")
    print("baseline (per-factor or fused).")
    if claimed:
        for r in claimed:
            print(f"  batch {r['batch']:>3}, length {r['length']:>5}: "
                  f"{r['claim_ratio']:.2f} of the best 4-factor time")
    else:
        print("  NO CELL meets it: no speedup is claimed.")
    print("Allocation 's5mix' is synthetic (order 4 w.p. 0.1, else 1), matching what")
    print("the learned gate reached in 13/20 runs; weights are random.")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(dict(
        device=torch.cuda.get_device_name(), K=K, rows=rows), indent=2))
    print(f"written: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="correctness checks only (Part 1)")
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--lengths", default="128,512,2048")
    ap.add_argument("--out", default="results/ragged_bench.json")
    args = ap.parse_args()
    ok = correctness()
    if args.selftest:
        sys.exit(0 if ok else 1)
    if not ok:
        print("\nPart 1 failed: timing an incorrect path means nothing. Stopping.")
        sys.exit(1)
    timing([int(b) for b in args.batches.split(",")],
           [int(t) for t in args.lengths.split(",")], args.out)


if __name__ == "__main__":
    main()
