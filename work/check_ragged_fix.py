"""
The faster S5 path loses more sequences than the normal path: is it our code, and can a
different kernel fix it?

    python work/check_ragged_fix.py --selftest
    python -u work/check_ragged_fix.py --ckpt "checkpoints/gate_B/*.pt" "checkpoints/gate_restart/*.pt" \
        > results/ragged_fix.log 2>&1

Pre-registered in PLAN.md ("Fixing the faster path's sequence accuracy", 25 October)
before any run.

BACKGROUND (results/seq_accuracy.json, 25 Oct): pooled over 20 trained S5 models, sequence
accuracy is 0.9891 on Path A (the training path), 0.9706 on the faster path R (fused
factors + ragged expansion, bf16 chunk kernel) and 0.9956 with the recurrence computed
exactly (X). The faster path loses ~4x more sequences to rounding than Path A.

SUSPECTS ON OUR SIDE, each tested here on the real trained models
  1. packing / indexing: R's maths is proven in float64 on small CPU cases only. Here the
     packed recurrence of the real model is run with an EXACT float64 kernel that honours
     cu_seqlens (path R64). R64 must reproduce X.
  2. the fused projections (our own concatenated short convolution): path F (fused Path A)
     must reproduce Path A.
  3. gate decisions flipping between paths (a discrete change, not rounding): the fraction
     of tokens whose number of factors differs from Path A's, per layer, for every path.
  4. chunk density: fla's chunk kernel solves a 64x64 triangular system per chunk and
     stores the inverse in bf16 (solve_tril, output_dtype=k.dtype). In Path A ~2/3 of the
     micro-steps in a chunk are closed factors (beta 0); after packing every micro-step is
     a real update, often near a reflection (beta ~ 2). Measured: the error of each
     path's recurrence output against a float64 recurrence on the SAME inputs, per layer
     and per position.
CANDIDATE KERNELS for the packed recurrence (the layer's kernel is swapped for the call):
  R16   the same chunk kernel with float16 inputs (10-bit mantissa instead of bf16's 7)
  Rrec  fla's fused_recurrent_gated_delta_rule: sequential, state kept in float32,
        accepts cu_seqlens
PATHS SCORED on each model's full test set (2 000 sequences):
  A  per-factor Path A (training path)   F  fused Path A        X  Path A, exact float64
  R  fused ragged, bf16 chunk (as timed) R64 fused ragged, exact float64 (packed)
  R16, Rrec as above

DECISION (pooled over models that learned the task: Path A token accuracy >= 0.99),
written before any run
  Step 1, our side (decides first):
    PACKING BUG    agreement(R64, X) < 0.9999 on predictions, or |seq(R64) - seq(X)| > 0.002
    FUSION BUG     agreement(F, A) < 0.9999, or |seq(F) - seq(A)| > 0.002
    either -> stop: the fault is ours; find and fix it before anything else
  Step 2, only if step 1 passes: a candidate kernel is ACCURATE if
    seq(candidate) >= seq(A) - 0.002 and token(candidate) >= token(A) - 0.0001
  Step 3, timing (on one model, as in the 13 Oct speed claim: test sequences joined to
    32 x 512 and 32 x 2048, learned gate inside the forward, median of 30 after 5 warm-up):
    a candidate is ADOPTED if ACCURATE and its time / best fixed-4-factor time <= 0.90 at
    both sizes (the original rule: at least 10% faster). If several, the fastest.
    None adopted -> the S5 speed claim keeps its sequence-level caveat.
REPORTED EITHER WAY: per-model accuracies for every path, discordant sequence counts,
gate-flip fractions, and the per-layer, per-position recurrence error of A, R, R16, Rrec.

CHECKS PER MODEL (a model failing one is reported and left out of the decision):
  G0 the checkpoint reloads (Path A token accuracy within 0.002 of the training record)
  G1 both exact paths really are exact: on the profiled sequences, X's and R64's recurrence
     output is within 5e-3 (relative L2; bf16 output rounding) of a float64 recurrence on
     the same inputs, in every layer and position block. Without it, a swap that silently
     failed to apply would make R64 equal R and read as a packing bug.
  G2 G1 can fail: Path A's own recurrence error exceeds 5e-3 somewhere.
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "work"))

from ragged_expansion import reference_gated_delta, path_a_core  # noqa: E402  (CPU-safe)
from check_seq_accuracy import exact_gated_delta, seq_correct, discordant  # noqa: E402

LEARNED = 0.99
G1_TOL = 5e-3
AGREE_MIN, SEQ_SAME, SEQ_TOL, TOK_TOL, TIME_MAX = 0.9999, 0.002, 0.002, 0.0001, 0.90
CANDIDATES = ("R16", "Rrec")
POS_BUCKETS = 4


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------

def unpack_index(cu, N):
    """Packed positions 0..N-1 -> (sequence id, position within the sequence)."""
    lengths = cu[1:] - cu[:-1]
    sid = torch.repeat_interleave(torch.arange(len(lengths), device=cu.device), lengths)
    pos = torch.arange(N, device=cu.device) - cu[:-1][sid]
    return sid, pos, int(lengths.max())


def exact_packed(q, k, v, g, beta, cu):
    """Exact float64 recurrence on packed sequences (batch 1, boundaries in cu): the state
    restarts at every boundary. Implemented by unpacking into a zero-padded batch; padding
    sits after each sequence's end, so it cannot affect any real output."""
    assert q.shape[0] == 1, "packed input must have batch 1"
    N = q.shape[1]
    sid, pos, L = unpack_index(cu, N)
    S = len(cu) - 1

    def pad(x):
        out = x.new_zeros((S, L) + tuple(x.shape[2:]))
        out[sid, pos] = x[0]
        return out

    o = exact_gated_delta(pad(q), pad(k), pad(v), None if g is None else pad(g), pad(beta))
    return o[sid, pos].unsqueeze(0)


def make_kernel(kind, gated, orig, recurrent=None):
    """Drop-in for fla's chunk_gated_delta_rule (gated) or chunk_delta_rule (not gated).
    kind: 'exact' (float64; packed sequences honoured), 'fp16' (orig with float16 inputs),
    'recurrent' (fla's fused_recurrent_gated_delta_rule; for an ungated layer, g = 0)."""
    def kernel(q, k, v, g=None, beta=None, initial_state=None, output_final_state=False,
               cu_seqlens=None, use_qk_l2norm_in_kernel=False, **kw):
        if not gated:
            beta, g = (g if beta is None else beta), None
        if initial_state is not None or output_final_state:
            raise RuntimeError("swapped kernel: no initial or final state")
        if not use_qk_l2norm_in_kernel:
            raise RuntimeError("swapped kernel: L2-normalised convention only")
        if kind == "exact":
            o = exact_gated_delta(q, k, v, g, beta) if cu_seqlens is None else exact_packed(q, k, v, g, beta, cu_seqlens)
            return o.to(q.dtype), None
        if kind == "fp16":
            h = lambda t: t.to(torch.float16)
            args = dict(q=h(q), k=h(k), v=h(v), beta=h(beta), initial_state=None, output_final_state=False,
                        cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=True)
            if gated:
                args["g"] = g
            o = orig(**args)[0]
            return o.to(q.dtype), None
        if kind == "recurrent":
            gg = g if gated else q.new_zeros(q.shape[:3], dtype=torch.float32)
            o = recurrent(q=q, k=k, v=v, g=gg, beta=beta, initial_state=None, output_final_state=False,
                          cu_seqlens=cu_seqlens, use_qk_l2norm_in_kernel=True)[0]
            return o.to(q.dtype), None
        raise ValueError(kind)
    return kernel


@contextlib.contextmanager
def kernel_swap(module, kind, recurrent=None):
    """Swap the layer module's two kernel names; always restore. kind None = no swap."""
    if kind is None:
        yield
        return
    saved = (module.chunk_gated_delta_rule, module.chunk_delta_rule)
    module.chunk_gated_delta_rule = make_kernel(kind, True, saved[0], recurrent)
    module.chunk_delta_rule = make_kernel(kind, False, saved[1], recurrent)
    try:
        yield
    finally:
        module.chunk_gated_delta_rule, module.chunk_delta_rule = saved


# ---------------------------------------------------------------------------
# measurements and decisions
# ---------------------------------------------------------------------------

def flip_fraction(n_a, n_b, valid=None):
    """Fraction of tokens whose number of factors differs between two paths."""
    d = n_a != n_b
    if valid is not None:
        d = d[valid]
    return d.float().mean().item()


def position_errors(o, ref, buckets=POS_BUCKETS):
    """Relative L2 error of o against ref, per contiguous block of token positions.
    o, ref: (B, T, H, Dv)."""
    T = o.shape[1]
    edges = [round(i * T / buckets) for i in range(buckets + 1)]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        d = (o[:, a:b].double() - ref[:, a:b].double()).norm()
        out.append((d / ref[:, a:b].double().norm().clamp_min(1e-12)).item())
    return out


def exact_reference(c):
    """Path A's recurrence recomputed exactly (float64) from captured per-factor inputs."""
    H = c["num_heads"]

    def heads(t):
        B, T, _ = t.shape
        return t.double().reshape(B, T, H, -1)

    ks, vs = [heads(x) for x in c["ks"]], [heads(x) for x in c["vs"]]
    betas = [b.double() for b in c["betas"]]
    g = None if c["g"] is None else c["g"].double()
    return path_a_core(heads(c["q"]), ks, vs, betas, g,
                       lambda q_, k_, v_, g_, b_: exact_gated_delta(q_, k_, v_, g_, b_))


def step1(agree_r64_x, seq_r64, seq_x, agree_f_a, seq_f, seq_a):
    out = []
    if agree_r64_x < AGREE_MIN or abs(seq_r64 - seq_x) > SEQ_SAME + 1e-12:
        out.append("PACKING BUG")
    if agree_f_a < AGREE_MIN or abs(seq_f - seq_a) > SEQ_SAME + 1e-12:
        out.append("FUSION BUG")
    return out or ["OUR SIDE CLEAR"]


def accurate(seq_c, tok_c, seq_a, tok_a):
    return seq_c >= seq_a - SEQ_TOL - 1e-12 and tok_c >= tok_a - TOK_TOL - 1e-12


def adopt(acc, ratios):
    """acc: {name: bool}; ratios: {name: [ratio at each size]} -> adopted name or None."""
    ok = [n for n in acc if acc[n] and n in ratios and all(r <= TIME_MAX + 1e-12 for r in ratios[n])]
    return min(ok, key=lambda n: max(ratios[n])) if ok else None


def pooled(rows, path, key):
    n = sum(r["n_seq"] for r in rows)
    return sum(r[path][key] * r["n_seq"] for r in rows) / n


# ---------------------------------------------------------------------------
# self-test (CPU)
# ---------------------------------------------------------------------------

def selftest() -> bool:
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""), flush=True)

    print("check_ragged_fix self-test (CPU, float64)")
    gen = torch.Generator().manual_seed(1)
    H, Dk, Dv = 3, 4, 5
    lens = [5, 9, 1, 7]
    N = sum(lens)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()))
    r = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float64)
    q, k, v = r(1, N, H, Dk), r(1, N, H, Dk), r(1, N, H, Dv)
    beta = 2 * torch.rand(1, N, H, generator=gen, dtype=torch.float64)
    g = -torch.rand(1, N, H, generator=gen, dtype=torch.float64)

    print("P1  exact recurrence on packed sequences")
    for name, gg in (("with decay", g), ("without decay", None)):
        ref = reference_gated_delta(q, k, v, gg, beta, cu)
        err = (exact_packed(q, k, v, gg, beta, cu) - ref).abs().max().item()
        check(f"= the validated reference with sequence boundaries ({name})", err < 1e-12, f"max err {err:.1e}")
    whole = reference_gated_delta(q, k, v, g, beta, None)
    check("control: ignoring the boundaries gives a different result",
          (exact_packed(q, k, v, g, beta, cu) - whole).abs().max().item() > 1e-3)
    sid, pos, L = unpack_index(cu, N)
    check("unpack index: sequence ids, positions and the longest length",
          sid.tolist()[:6] == [0, 0, 0, 0, 0, 1] and pos.tolist()[5:8] == [0, 1, 2] and L == 9)

    print("P2  swapped kernels: plumbing")
    seen = {}

    def fake_orig(**kw):
        seen.update(kw)
        return (kw["q"] * 1.0, None)

    def fake_rec(**kw):
        seen.update(kw)
        return (kw["v"][..., :1].expand(*kw["v"].shape[:3], Dv) * 0 + 1, None)

    qb, kb, vb, bb = (t.to(torch.bfloat16) for t in (q, k, v, beta))
    kx = make_kernel("exact", True, fake_orig)
    o, st = kx(q=qb, k=kb, v=vb, g=g.float(), beta=bb, cu_seqlens=cu, use_qk_l2norm_in_kernel=True)
    check("exact kernel honours cu_seqlens and returns the input dtype",
          st is None and o.dtype == torch.bfloat16
          and torch.equal(o, exact_packed(qb, kb, vb, g.float(), bb, cu).to(torch.bfloat16)))
    o, _ = kx(q=qb, k=kb, v=vb, g=g.float(), beta=bb, use_qk_l2norm_in_kernel=True)
    check("exact kernel without cu_seqlens = the batched exact recurrence",
          torch.equal(o, exact_gated_delta(qb, kb, vb, g.float(), bb).to(torch.bfloat16)))
    k16 = make_kernel("fp16", True, fake_orig)
    seen.clear()
    o, _ = k16(q=qb, k=kb, v=vb, g=g.float(), beta=bb, cu_seqlens=cu, use_qk_l2norm_in_kernel=True)
    check("fp16 kernel: the original kernel receives float16 q, k, v, beta, the same g and cu_seqlens; output back in bf16",
          seen["q"].dtype == seen["k"].dtype == seen["v"].dtype == seen["beta"].dtype == torch.float16
          and seen["g"].dtype == torch.float32 and seen["cu_seqlens"] is cu and o.dtype == torch.bfloat16)
    k16u = make_kernel("fp16", False, fake_orig)
    seen.clear()
    k16u(q=qb, k=kb, v=vb, beta=bb, use_qk_l2norm_in_kernel=True)
    check("fp16 kernel, ungated layer: no g is passed", "g" not in seen)
    kr = make_kernel("recurrent", False, fake_orig, fake_rec)
    seen.clear()
    kr(q=qb, k=kb, v=vb, beta=bb, cu_seqlens=cu, use_qk_l2norm_in_kernel=True)
    check("recurrent kernel, ungated layer: g = 0 (no decay), cu_seqlens and L2 norm passed on",
          torch.equal(seen["g"], torch.zeros(1, N, H)) and seen["cu_seqlens"] is cu and seen["use_qk_l2norm_in_kernel"] is True
          and torch.equal(seen["beta"], bb))
    refused = []
    for kw in (dict(initial_state=torch.zeros(1)), dict(output_final_state=True), dict(use_qk_l2norm_in_kernel=False)):
        a = dict(q=qb, k=kb, v=vb, g=g.float(), beta=bb, use_qk_l2norm_in_kernel=True)
        a.update(kw)
        try:
            kx(**a)
            refused.append(False)
        except RuntimeError:
            refused.append(True)
    check("refuses initial state, final state and the un-normalised convention", all(refused))

    print("P3  the swap is applied and always undone")
    class Mod:
        chunk_gated_delta_rule = staticmethod(fake_orig)
        chunk_delta_rule = staticmethod(fake_orig)
    m = Mod()
    og = m.chunk_gated_delta_rule
    with kernel_swap(m, "exact"):
        swapped = m.chunk_gated_delta_rule is not og
    check("swapped inside, restored after", swapped and m.chunk_gated_delta_rule is og)
    try:
        with kernel_swap(m, "fp16"):
            raise ValueError
    except ValueError:
        pass
    check("restored after an exception", m.chunk_gated_delta_rule is og)
    with kernel_swap(m, None):
        check("kind None leaves the kernel alone", m.chunk_gated_delta_rule is og)

    print("P3b exact reference from captured per-factor inputs")
    Bc, Tc, Hc, Dc, Kc = 2, 6, 2, 3, 3
    mk = lambda *sh: torch.randn(*sh, generator=gen, dtype=torch.float64)
    cap = {"num_heads": Hc, "q": mk(Bc, Tc, Hc * Dc), "ks": [mk(Bc, Tc, Hc * Dc) for _ in range(Kc)],
           "vs": [mk(Bc, Tc, Hc * Dc) for _ in range(Kc)],
           "betas": [2 * torch.rand(Bc, Tc, Hc, generator=gen, dtype=torch.float64) for _ in range(Kc)],
           "g": -torch.rand(Bc, Tc, Hc, generator=gen, dtype=torch.float64)}
    hd = lambda t: t.reshape(Bc, Tc, Hc, -1)
    want = path_a_core(hd(cap["q"]), [hd(x) for x in cap["ks"]], [hd(x) for x in cap["vs"]], cap["betas"], cap["g"],
                       lambda q_, k_, v_, g_, b_: reference_gated_delta(q_, k_, v_, g_, b_, None))
    err = (exact_reference(cap) - want).abs().max().item()
    check("= Path A's maths via the validated sequential reference (with decay)", err < 1e-12, f"max err {err:.1e}")

    print("P4  measurements and decisions on known answers")
    check("gate flips: 1 of 4 tokens differ", flip_fraction(torch.tensor([1, 2, 2, 1]), torch.tensor([1, 2, 1, 1])) == 0.25)
    ref4 = torch.ones(1, 8, 1, 1, dtype=torch.float64)
    o4 = ref4.clone()
    o4[:, 6:] *= 1.1
    pe = position_errors(o4, ref4)
    check("position errors: only the last block is off, by 10%", [round(x, 6) for x in pe] == [0, 0, 0, 0.1], str(pe))
    check("step 1 clear", step1(0.99995, 0.990, 0.991, 1.0, 0.9890, 0.9891) == ["OUR SIDE CLEAR"])
    check("step 1: low R64/X agreement -> PACKING BUG", step1(0.9990, 0.990, 0.990, 1.0, 0.99, 0.99) == ["PACKING BUG"])
    check("step 1: R64 sequence accuracy off by 0.003 -> PACKING BUG", step1(0.99995, 0.987, 0.990, 1.0, 0.99, 0.99) == ["PACKING BUG"])
    check("step 1: fused path differs -> FUSION BUG", step1(1.0, 0.99, 0.99, 0.9990, 0.99, 0.99) == ["FUSION BUG"])
    check("accurate: within 0.002 of Path A (inclusive), token within 0.0001",
          accurate(0.987, 0.9998, 0.989, 0.9998) and not accurate(0.9869, 0.9998, 0.989, 0.9998)
          and not accurate(0.990, 0.99968, 0.989, 0.9998))
    check("adopt: accurate and <= 0.90 at both sizes; fastest of several",
          adopt({"R16": True, "Rrec": True}, {"R16": [0.80, 0.85], "Rrec": [0.70, 0.95]}) == "R16"
          and adopt({"R16": False, "Rrec": True}, {"R16": [0.5, 0.5], "Rrec": [0.88, 0.90]}) == "Rrec"
          and adopt({"R16": True}, {"R16": [0.91, 0.5]}) is None and adopt({"R16": True}, {"R16": [0.5, 0.91]}) is None)
    rows = [{"n_seq": 100, "A": {"seq_acc": 1.0}}, {"n_seq": 300, "A": {"seq_acc": 0.0}}]
    check("pooling weights by sequences (0.25)", abs(pooled(rows, "A", "seq_acc") - 0.25) < 1e-12)
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return ok


# ---------------------------------------------------------------------------
# GPU run
# ---------------------------------------------------------------------------

PATHS = (  # name, fused, ragged, kernel swap
    ("A", False, False, None), ("F", True, False, None), ("X", False, False, "exact"),
    ("R", True, True, None), ("R64", True, True, "exact"), ("R16", True, True, "fp16"), ("Rrec", True, True, "recurrent"),
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt", nargs="*", default=[])
    ap.add_argument("--timing_ckpt", default="learned_g_0_1_hard_1_g0.1_seed12.pt")
    ap.add_argument("--n_loc", type=int, default=16, help="test sequences for the per-position error profile")
    ap.add_argument("--out", default=str(ROOT / "results" / "ragged_fix.json"))
    args = ap.parse_args()
    if not selftest():
        sys.exit(1)
    if args.selftest:
        return
    paths = sorted({p for pat in args.ckpt for p in glob.glob(pat)})
    if not paths:
        sys.exit("no checkpoint found")

    import fla.layers.gated_deltaproduct as gdp
    from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule as fr
    from exp_matched_compute import build, load, DEV
    from exp_gate_train import set_gate_hard, collect_n_t
    from check_ragged import set_fused, set_ragged, adaptive_layers, time_fn
    from check_ragged_trained import data_csv

    def run_path(model, te_x, fused, ragged, swap):
        set_fused(model, fused)
        set_ragged(model, ragged)
        bs = 32 if swap == "exact" else 128
        preds, nts = [], []
        with kernel_swap(gdp, swap, fr), torch.inference_mode():
            for i in range(0, te_x.shape[0], bs):
                preds.append(model(input_ids=te_x[i:i + bs], use_cache=False).logits.argmax(-1))
                nts.append(torch.stack(collect_n_t(model)).cpu())
        set_fused(model, False)
        set_ragged(model, False)
        return torch.cat(preds), torch.cat(nts, 1)        # (N, T), (layers, N, T)

    def localise(model, xs, fused, ragged, swap):
        """Per layer, per position block: recurrence-output error against an exact
        recurrence on the same captured inputs."""
        layers = adaptive_layers(model)
        for L in layers:
            L.capture = []
        set_fused(model, fused)
        set_ragged(model, ragged)
        try:
            with kernel_swap(gdp, swap, fr), torch.inference_mode():
                model(input_ids=xs, use_cache=False)
            caps = [L.capture[0] for L in layers]
        finally:
            for L in layers:
                L.capture = None
            set_fused(model, False)
            set_ragged(model, False)
        return [position_errors(c["o"], exact_reference(c)) for c in caps]

    rows, timing = [], None
    data_cache = {}
    for path in paths:
        ck = torch.load(path, map_location=DEV, weights_only=True)
        b, d, res = ck["build"], ck["data"], ck["result"]
        if not b["adaptive"] or res.get("token_acc", 0) < LEARNED:
            print(f"{Path(path).name}: skipped (not adaptive, or recorded token accuracy {res.get('token_acc', 0):.4f} < {LEARNED})")
            continue
        key = (str(data_csv(d)), d["n_train"], d["n_test"])
        if key not in data_cache:
            data_cache[key] = load(data_csv(d), d["n_train"], d["n_test"])
        _, _, te_x, te_y = data_cache[key]
        model = build(**b)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        set_gate_hard(model, True)
        # Pinned 25 Oct: these paths are defined with the packed recurrence on the bf16 chunk
        # kernel and the other kernels swapped in for it. The layer's default became the
        # recurrent kernel after this script's run; pin the old one so the labels keep their
        # pre-registered meaning.
        for _m in model.modules():
            if hasattr(_m, "ragged_kernel"):
                _m.ragged_kernel = "chunk"
        name = Path(path).name
        print("=" * 78 + f"\n{name}   trained: token {res['token_acc']:.5f}", flush=True)
        row = {"checkpoint": name, "n_seq": int(te_x.shape[0])}
        out = {}
        for label, fused, ragged, swap in PATHS:
            p, nt = run_path(model, te_x, fused, ragged, swap)
            ok = seq_correct(p, te_y)
            out[label] = (p, nt, ok)
            row[label] = {"token_acc": (p == te_y).float().mean().item(), "seq_acc": ok.float().mean().item()}
        g0 = abs(row["A"]["token_acc"] - res["token_acc"]) <= 0.002
        row["g0"] = g0
        print(f"  [{'PASS' if g0 else 'FAIL'}] G0 reload: Path A token {row['A']['token_acc']:.5f}, recorded {res['token_acc']:.5f}")
        pa, na, oka = out["A"]
        for label in (p_[0] for p_ in PATHS[1:]):
            p, nt, okp = out[label]
            row[label]["agree_A"] = (p == pa).float().mean().item()
            row[label]["disc_vs_A"] = discordant(oka, okp)
            row[label]["gate_flips_vs_A"] = [flip_fraction(na[l], nt[l]) for l in range(na.shape[0])]
        row["agree_R64_X"] = (out["R64"][0] == out["X"][0]).float().mean().item()
        print("  " + "  ".join(f"{l} {row[l]['seq_acc']:.4f}" for l, *_ in PATHS) + "   (sequence accuracy)")
        print("  " + "  ".join(f"{l} {row[l]['token_acc']:.5f}" for l, *_ in PATHS) + "   (token accuracy)")
        print(f"  agreement with A: " + "  ".join(f"{l} {row[l]['agree_A']:.5f}" for l, *_ in PATHS[1:])
              + f"   R64 with X {row['agree_R64_X']:.5f}")
        print(f"  gate flips vs A (per layer): " + "  ".join(f"{l} {row[l]['gate_flips_vs_A']}" for l, *_ in PATHS[1:]))
        xs = te_x[:args.n_loc]
        row["pos_error"] = {lab: localise(model, xs, f, r, s) for lab, f, r, s in
                            (("A", False, False, None), ("R", True, True, None), ("R16", True, True, "fp16"),
                             ("Rrec", True, True, "recurrent"), ("X", False, False, "exact"), ("R64", True, True, "exact"))}
        mx = lambda lab: max(max(e) for e in row["pos_error"][lab])
        row["g1"] = mx("X") <= G1_TOL and mx("R64") <= G1_TOL
        row["g2"] = mx("A") > G1_TOL
        print(f"  [{'PASS' if row['g1'] else 'FAIL'}] G1 exact paths are exact: max error X {mx('X'):.1e}, R64 {mx('R64'):.1e}")
        print(f"  [{'PASS' if row['g2'] else 'FAIL'}] G2 G1 can fail: Path A max error {mx('A'):.1e} > {G1_TOL:.0e}")
        for lab, per_layer in row["pos_error"].items():
            print(f"  recurrence error vs exact, {lab:>4}: " + " | ".join(
                "layer %d: %s" % (i, " ".join(f"{e:.1e}" for e in errs)) for i, errs in enumerate(per_layer)))
        rows.append(row)

        if name == args.timing_ckpt:
            fixed = build(n_h=4, adaptive=False, n_heads=b["n_heads"], head_dim=b["head_dim"], hidden=b["hidden"],
                          n_layers=b["n_layers"], vocab=b["vocab"], seed=0).eval()
            timing = {"checkpoint": name, "sizes": []}
            for B, n_join in ((32, 4), (32, 16)):
                x = te_x[:B * n_join].reshape(B, n_join * te_x.shape[1])
                fwd = lambda m: (lambda: m(input_ids=x, use_cache=False))
                set_fused(fixed, False); t4 = time_fn(fwd(fixed))
                set_fused(fixed, True); t4f = time_fn(fwd(fixed)); set_fused(fixed, False)
                best4 = min(t4, t4f)
                entry = {"batch": B, "length": int(x.shape[1]), "best4_ms": best4}
                set_fused(model, True); set_ragged(model, True)
                for lab, swap in (("R", None), ("R16", "fp16"), ("Rrec", "recurrent")):
                    with kernel_swap(gdp, swap, fr):
                        t = time_fn(fwd(model))
                    entry[f"{lab}_ms"], entry[f"{lab}_ratio"] = t, t / best4
                set_fused(model, False); set_ragged(model, False)
                timing["sizes"].append(entry)
                print(f"  timing {B} x {x.shape[1]}: best 4-factor {best4:.2f} ms; " + "  ".join(
                    f"{lab} {entry[lab + '_ms']:.2f} ms ({entry[lab + '_ratio']:.2f})" for lab in ("R", "R16", "Rrec")), flush=True)

    good = [r for r in rows if r["g0"] and r["g1"] and r["g2"] and r["A"]["token_acc"] >= LEARNED]
    summary = {"n_models": len(good)}
    print("\n" + "=" * 78)
    if good:
        P = {l: {k: pooled(good, l, k) for k in ("seq_acc", "token_acc")} for l, *_ in PATHS}
        agree = lambda key_fn: sum(key_fn(r) * r["n_seq"] for r in good) / sum(r["n_seq"] for r in good)
        a_r64x = agree(lambda r: r["agree_R64_X"])
        a_fa = agree(lambda r: r["F"]["agree_A"])
        s1 = step1(a_r64x, P["R64"]["seq_acc"], P["X"]["seq_acc"], a_fa, P["F"]["seq_acc"], P["A"]["seq_acc"])
        acc = {c: accurate(P[c]["seq_acc"], P[c]["token_acc"], P["A"]["seq_acc"], P["A"]["token_acc"]) for c in CANDIDATES}
        ratios = {c: [s[f"{c}_ratio"] for s in timing["sizes"]] for c in CANDIDATES} if timing else {}
        adopted = adopt(acc, ratios) if s1 == ["OUR SIDE CLEAR"] else None
        summary.update(pooled=P, agree_R64_X=a_r64x, agree_F_A=a_fa, step1=s1, accurate=acc, ratios=ratios,
                       adopted=adopted, timing=timing)
        print(f"POOLED over {len(good)} models:")
        print("  sequence " + "  ".join(f"{l} {P[l]['seq_acc']:.4f}" for l, *_ in PATHS))
        print("  token    " + "  ".join(f"{l} {P[l]['token_acc']:.5f}" for l, *_ in PATHS))
        print(f"  agreement R64 with X {a_r64x:.6f}; F with A {a_fa:.6f}")
        print(f"  STEP 1 (our side): {', '.join(s1)}")
        if s1 == ["OUR SIDE CLEAR"]:
            print("  STEP 2 (accurate): " + "  ".join(f"{c} {'yes' if acc[c] else 'no'}" for c in CANDIDATES))
            print("  STEP 3 (time / best 4-factor at 32x512, 32x2048): " + "  ".join(
                f"{c} {ratios.get(c)}" for c in CANDIDATES))
            print(f"  ADOPTED: {adopted or 'none -> the S5 speed claim keeps its sequence-level caveat'}")
        else:
            print("  stop: the fault is on our side; candidates are not judged")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"rows": rows, "summary": summary}, f, indent=1, default=float)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
