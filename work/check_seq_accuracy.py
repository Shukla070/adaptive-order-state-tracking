"""
Why does sequence accuracy drop on the faster (ragged) path? A third, exact path decides.

    python work/check_seq_accuracy.py --selftest
    python -u work/check_seq_accuracy.py --ckpt "checkpoints/gate_B/*.pt" "checkpoints/gate_restart/*.pt" \
        > results/seq_accuracy.log 2>&1

Pre-registered in PLAN.md ("Sequence accuracy on the faster path", 25 October) before any run.

THE QUESTION
------------
On trained S5 models the faster path keeps token accuracy (0.99979 -> 0.99959 on seed 12)
but whole-sequence accuracy drops (0.9795 -> 0.9595). The two paths compute the same
mathematics; they differ only in bf16 rounding inside the Triton kernel. Either
  (a) the faster path has a flaw of its own -- our fault, to be fixed; or
  (b) the model's sequence accuracy is sensitive to ANY change in recurrence rounding,
      and the normal path (Path A) looks better only because the model was trained
      through its exact rounding.
Earlier evidence pointed both ways: both paths are equally far from a float64 recurrence
(relative error 0.0188 vs 0.0181), but the tokens where they disagree are not near-ties
(median top-1 minus top-2 logit margin 2.3, against 7.8 overall).

THE THIRD PATH (X)
------------------
The same model and weights, with only the recurrence computed exactly: the layer's
`chunk_gated_delta_rule` / `chunk_delta_rule` are swapped, for the duration of the call,
for a float64 sequential recurrence with fla's convention (L2-normalised q and k,
S <- S exp(g); S <- S + beta k (v - S^T k)^T; o = S^T q / sqrt(Dk)), on the same bf16
inputs, with the output rounded once to bf16. Everything else in the model is unchanged.
The convention is the one `src/ragged_expansion.reference_gated_delta` implements and
work/check_ragged.py validated against the kernel (C1b).

PATHS SCORED on the full test set of each checkpoint (128-token sequences):
  A   Path A, per-factor -- the path the model was trained with
  R   the faster path as timed: fused factors + ragged expansion
  X   Path A with the exact float64 recurrence

DECISION (pooled over the models that learned the task: Path A token accuracy >= 0.99),
written before any run:
  NO GENERAL DROP        pooled seq(A) - seq(R) <= 0.005: the seed-12 drop does not
                         generalise; the per-model range is reported with the claim.
  otherwise, attribution by X:
  FASTER-PATH FAULT      seq(X) >= seq(R) + 0.01 and |seq(X) - seq(A)| <= 0.005:
                         the exact recurrence agrees with Path A; fix the faster path.
  ROUNDING SENSITIVITY   seq(X) <= seq(A) - 0.01: exact arithmetic also loses sequences;
                         the faster path is not at fault, and the claim states that
                         sequence accuracy moves by up to this much between numerically
                         equivalent paths.
  UNCLEAR                anything else.
Reported either way: per-model accuracies, sequence-level discordant counts for each pair
of paths, and logit margins at the tokens where paths disagree.

CHECKS ON THE GPU, before any scoring (each model):
  G0  the checkpoint reloads: Path A token accuracy within 0.002 of the training record
  G1  the swap is active and exact: on 8 test sequences the X path's recurrence output
      is within 5e-3 (relative L2; bf16 output rounding) of the validated float64
      reference, and at most half of Path A's own error against it
  G2  G1 must be able to fail: Path A's own error must exceed G1's tolerance (otherwise
      the tolerance cannot tell an exact path from Path A)
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

from ragged_expansion import reference_gated_delta  # noqa: E402  (float64, CPU-safe)

LEARNED_TOKEN_ACC = 0.99
NO_DROP, ATTR, SAME = 0.005, 0.01, 0.005
G1_TOL = 5e-3


# ---------------------------------------------------------------------------
# the exact recurrence
# ---------------------------------------------------------------------------

def exact_gated_delta(q, k, v, g, beta):
    """Sequential gated delta rule in float64, vectorised over batch and heads.
    q, k (B, L, H, Dk); v (B, L, H, Dv); g (B, L, H) or None; beta (B, L, H).
    Same convention as fla's chunk kernels with use_qk_l2norm_in_kernel=True."""
    q, k, v, beta = (t.double() for t in (q, k, v, beta))
    g = None if g is None else g.double()
    B, L, H, Dk = q.shape
    scale = Dk ** -0.5
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    S = q.new_zeros(B, H, Dk, v.shape[-1])
    out = q.new_zeros(B, L, H, v.shape[-1])
    for t in range(L):
        if g is not None:
            S = S * torch.exp(g[:, t]).view(B, H, 1, 1)
        kt, vt, bt = k[:, t], v[:, t], beta[:, t]
        v_old = torch.einsum("bhk,bhkv->bhv", kt, S)
        S = S + bt.view(B, H, 1, 1) * torch.einsum("bhk,bhv->bhkv", kt, vt - v_old)
        out[:, t] = torch.einsum("bhk,bhkv->bhv", q[:, t], S) * scale
    return out


def make_exact_kernel(gated: bool):
    """A drop-in for fla's chunk_gated_delta_rule (gated=True) or chunk_delta_rule:
    same call signature, exact float64 recurrence, output rounded once to q's dtype.
    Refuses the features Path A does not use here (cache, initial state, packing)."""
    def kernel(q, k, v, g=None, beta=None, initial_state=None, output_final_state=False,
               cu_seqlens=None, use_qk_l2norm_in_kernel=False, **kw):
        if not gated:
            beta, g = g if beta is None else beta, None
        if initial_state is not None or output_final_state or cu_seqlens is not None:
            raise RuntimeError("exact kernel: no initial state, final state or cu_seqlens")
        if not use_qk_l2norm_in_kernel:
            raise RuntimeError("exact kernel implements the L2-normalised convention only")
        o = exact_gated_delta(q, k, v, g if gated else None, beta)
        return o.to(q.dtype), None
    return kernel


@contextlib.contextmanager
def exact_recurrence(module):
    """Swap the module's two kernel names for exact ones; always restore."""
    saved = (module.chunk_gated_delta_rule, module.chunk_delta_rule)
    module.chunk_gated_delta_rule = make_exact_kernel(True)
    module.chunk_delta_rule = make_exact_kernel(False)
    try:
        yield
    finally:
        module.chunk_gated_delta_rule, module.chunk_delta_rule = saved


# ---------------------------------------------------------------------------
# scoring and the decision
# ---------------------------------------------------------------------------

def seq_correct(pred, tgt):
    """(N, T) predictions and targets -> (N,) bool: every token right."""
    return (pred == tgt).all(-1)


def discordant(a_ok, b_ok):
    """Sequence-level discordance between two paths: (a right & b wrong, a wrong & b right)."""
    return int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())


def decide(sA, sR, sX):
    if sA - sR <= NO_DROP + 1e-12:
        return "NO GENERAL DROP"
    if sX >= sR + ATTR - 1e-12 and abs(sX - sA) <= SAME + 1e-12:
        return "FASTER-PATH FAULT"
    if sX <= sA - ATTR + 1e-12:
        return "ROUNDING SENSITIVITY"
    return "UNCLEAR"


def pooled(rows, key):
    n = sum(r["n_seq"] for r in rows)
    return sum(r[key]["seq_acc"] * r["n_seq"] for r in rows) / n if n else float("nan")


# ---------------------------------------------------------------------------
# self-test (CPU, no fla needed)
# ---------------------------------------------------------------------------

def selftest() -> bool:
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""), flush=True)

    print("check_seq_accuracy self-test (CPU, float64)")
    gen = torch.Generator().manual_seed(0)
    B, L, H, Dk, Dv = 2, 13, 3, 4, 5
    q, k = torch.randn(B, L, H, Dk, generator=gen, dtype=torch.float64), torch.randn(B, L, H, Dk, generator=gen, dtype=torch.float64)
    v = torch.randn(B, L, H, Dv, generator=gen, dtype=torch.float64)
    beta = 2 * torch.rand(B, L, H, generator=gen, dtype=torch.float64)
    g = -torch.rand(B, L, H, generator=gen, dtype=torch.float64)
    for name, gg in (("with decay", g), ("without decay", None)):
        ref = reference_gated_delta(q, k, v, gg, beta)
        err = (exact_gated_delta(q, k, v, gg, beta) - ref).abs().max().item()
        check(f"X1 exact recurrence = the validated reference ({name})", err < 1e-12, f"max err {err:.1e}")
    ref = reference_gated_delta(q, k, v, g, beta)
    check("X1b control: dropping the decay changes the result",
          (reference_gated_delta(q, k, v, None, beta) - ref).abs().max().item() > 1e-3)

    print("X2  drop-in kernel interface")
    kg, kd = make_exact_kernel(True), make_exact_kernel(False)
    qb, kb_, vb, bb = (t.to(torch.bfloat16) for t in (q, k, v, beta))
    o, st = kg(q=qb, k=kb_, v=vb, g=g.float(), beta=bb, initial_state=None, output_final_state=False,
               cu_seqlens=None, use_qk_l2norm_in_kernel=True)
    check("gated kernel returns (o in the input dtype, None)", o.dtype == torch.bfloat16 and st is None and o.shape == (B, L, H, Dv))
    exact_on_rounded = exact_gated_delta(qb, kb_, vb, g.float(), bb)
    check("output = exact recurrence on the bf16 inputs, rounded once",
          torch.equal(o, exact_on_rounded.to(torch.bfloat16)))
    o2, _ = kd(q=qb, k=kb_, v=vb, beta=bb, initial_state=None, output_final_state=False,
               cu_seqlens=None, use_qk_l2norm_in_kernel=True)
    check("ungated kernel = exact recurrence without decay",
          torch.equal(o2, exact_gated_delta(qb, kb_, vb, None, bb).to(torch.bfloat16)))
    raised = []
    for kw in (dict(initial_state=torch.zeros(1)), dict(output_final_state=True),
               dict(cu_seqlens=torch.tensor([0, L])), dict(use_qk_l2norm_in_kernel=False)):
        args = dict(q=qb, k=kb_, v=vb, g=g.float(), beta=bb, initial_state=None, output_final_state=False,
                    cu_seqlens=None, use_qk_l2norm_in_kernel=True)
        args.update(kw)
        try:
            kg(**args)
            raised.append(False)
        except RuntimeError:
            raised.append(True)
    check("refuses initial state, final state, packing, and the un-normalised convention", all(raised), str(raised))

    print("X3  the swap is applied and always undone")
    class Mod:  # stands in for fla.layers.gated_deltaproduct
        chunk_gated_delta_rule = staticmethod(lambda *a, **k: "orig_g")
        chunk_delta_rule = staticmethod(lambda *a, **k: "orig_d")
    m = Mod()
    og, od = m.chunk_gated_delta_rule, m.chunk_delta_rule
    with exact_recurrence(m):
        swapped = m.chunk_gated_delta_rule is not og and m.chunk_delta_rule is not od
    check("swapped inside, restored after", swapped and m.chunk_gated_delta_rule is og and m.chunk_delta_rule is od)
    try:
        with exact_recurrence(m):
            raise ValueError
    except ValueError:
        pass
    check("restored after an exception too", m.chunk_gated_delta_rule is og)

    print("X4  scoring and the decision on known answers")
    pred = torch.tensor([[1, 2, 3], [1, 2, 0], [4, 4, 4]])
    tgt = torch.tensor([[1, 2, 3], [1, 2, 3], [4, 4, 0]])
    check("sequence correct only if every token is", seq_correct(pred, tgt).tolist() == [True, False, False])
    a_ok, b_ok = torch.tensor([True, True, False, False]), torch.tensor([True, False, True, False])
    check("discordant counts (a right & b wrong, a wrong & b right) = (1, 1)", discordant(a_ok, b_ok) == (1, 1))
    check("NO GENERAL DROP at a pooled drop of 0.005 (inclusive)", decide(0.98, 0.975, 0.90) == "NO GENERAL DROP")
    check("FASTER-PATH FAULT: X with A, both above R", decide(0.98, 0.96, 0.978) == "FASTER-PATH FAULT")
    check("ROUNDING SENSITIVITY: X also loses", decide(0.98, 0.96, 0.965) == "ROUNDING SENSITIVITY")
    check("UNCLEAR: X between, closer to neither rule", decide(0.98, 0.96, 0.972) == "UNCLEAR")
    rows = [{"n_seq": 100, "A": {"seq_acc": 1.0}}, {"n_seq": 300, "A": {"seq_acc": 0.0}}]
    check("pooling weights models by their number of sequences (0.25)", abs(pooled(rows, "A") - 0.25) < 1e-12)
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return ok


# ---------------------------------------------------------------------------
# GPU run
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt", nargs="*", default=[])
    ap.add_argument("--out", default=str(ROOT / "results" / "seq_accuracy.json"))
    args = ap.parse_args()
    if not selftest():
        sys.exit(1)
    if args.selftest:
        return
    paths = sorted({p for pat in args.ckpt for p in glob.glob(pat)})
    if not paths:
        sys.exit("no checkpoint found; pass --ckpt with the saved S5 models")

    import fla.layers.gated_deltaproduct as gdp
    from exp_matched_compute import build, load, DEV
    from exp_gate_train import set_gate_hard
    from check_ragged import set_fused, set_ragged, adaptive_layers, reference_output, rel_err
    from check_ragged_trained import data_csv

    def predict(model, te_x, fused, ragged, exact):
        set_fused(model, fused)
        set_ragged(model, ragged)
        preds, margins = [], []
        ctx = exact_recurrence(gdp) if exact else contextlib.nullcontext()
        bs = 32 if exact else 128      # the float64 path holds (bs, T*K, H, Dv) doubles
        with ctx, torch.inference_mode():
            for i in range(0, te_x.shape[0], bs):
                lg = model(input_ids=te_x[i:i + bs], use_cache=False).logits.float()
                top2 = lg.topk(2, dim=-1).values
                preds.append(lg.argmax(-1))
                margins.append(top2[..., 0] - top2[..., 1])
        set_fused(model, False)
        set_ragged(model, False)
        return torch.cat(preds), torch.cat(margins)

    def captured_error(model, xs, exact):
        """Relative error of the first adaptive layer's recurrence output against the
        validated float64 reference, with or without the swap."""
        layer = adaptive_layers(model)[0]
        layer.capture = []
        ctx = exact_recurrence(gdp) if exact else contextlib.nullcontext()
        try:
            with ctx, torch.inference_mode():
                model(input_ids=xs, use_cache=False)
            c = layer.capture[0]
        finally:
            layer.capture = None
        return rel_err(c["o"], reference_output(c))

    data_cache, rows, all_ok = {}, [], True
    for path in paths:
        ck = torch.load(path, map_location=DEV, weights_only=True)
        b, d, res = ck["build"], ck["data"], ck["result"]
        if not b["adaptive"]:
            print(f"{Path(path).name}: not adaptive, skipped")
            continue
        key = (str(data_csv(d)), d["n_train"], d["n_test"])
        if key not in data_cache:
            data_cache[key] = load(data_csv(d), d["n_train"], d["n_test"])
        _, _, te_x, te_y = data_cache[key]
        model = build(**b)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        set_gate_hard(model, True)
        print("=" * 78)
        print(f"{Path(path).name}   trained: token {res.get('token_acc', float('nan')):.4f}, "
              f"order {res.get('mean_n_t', float('nan')):.4f}")
        ok = True

        def check(name, cond, detail=""):
            nonlocal ok
            ok &= bool(cond)
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}   {detail}", flush=True)

        err_x = captured_error(model, te_x[:8], exact=True)
        err_a = captured_error(model, te_x[:8], exact=False)
        check("G1 the swap is active and exact", err_x <= G1_TOL and err_x <= 0.5 * err_a,
              f"X {err_x:.2e}, Path A {err_a:.2e}")
        check("G2 G1 can fail: Path A's own error exceeds G1's tolerance", err_a > G1_TOL,
              f"Path A {err_a:.2e} > {G1_TOL:.0e}")
        row = {"checkpoint": Path(path).name, "trained": res, "n_seq": int(te_x.shape[0]),
               "g1_err_x": err_x, "g1_err_a": err_a}
        out = {}
        for label, fused, ragged, exact in (("A", False, False, False), ("R", True, True, False), ("X", False, False, True)):
            p, mg = predict(model, te_x, fused, ragged, exact)
            out[label] = (p, mg, seq_correct(p, te_y))
            row[label] = {"token_acc": (p == te_y).float().mean().item(),
                          "seq_acc": seq_correct(p, te_y).float().mean().item()}
        check("G0 checkpoint reloads: Path A token accuracy matches the training record",
              abs(row["A"]["token_acc"] - res.get("token_acc", float("nan"))) <= 0.002,
              f"now {row['A']['token_acc']:.5f}, recorded {res.get('token_acc', float('nan')):.5f}")
        for a, bb in (("A", "R"), ("A", "X"), ("R", "X")):
            row[f"disc_{a}{bb}"] = discordant(out[a][2], out[bb][2])
            dis = out[a][0] != out[bb][0]
            row[f"tok_disagree_{a}{bb}"] = int(dis.sum())
            row[f"margin_disagree_{a}{bb}"] = out["A"][1][dis].median().item() if dis.any() else None
        row["margin_all"] = out["A"][1].median().item()
        row["checks_pass"] = ok
        all_ok &= ok
        print(f"  token  A {row['A']['token_acc']:.5f}  R {row['R']['token_acc']:.5f}  X {row['X']['token_acc']:.5f}")
        print(f"  seq    A {row['A']['seq_acc']:.4f}  R {row['R']['seq_acc']:.4f}  X {row['X']['seq_acc']:.4f}"
              f"   discordant sequences A/R {row['disc_AR']}  A/X {row['disc_AX']}  R/X {row['disc_RX']}")
        print(f"  Path A logit margin, median: all tokens {row['margin_all']:.3f}; where A and R disagree "
              f"{row['margin_disagree_AR']}; where A and X disagree {row['margin_disagree_AX']}")
        rows.append(row)

    learned = [r for r in rows if r["A"]["token_acc"] >= LEARNED_TOKEN_ACC and r["checks_pass"]]
    summary = {"n_models": len(rows), "n_learned": len(learned)}
    print("\n" + "=" * 78)
    if learned:
        sA, sR, sX = (pooled(learned, k) for k in ("A", "R", "X"))
        verdict = decide(sA, sR, sX)
        drops = sorted(r["A"]["seq_acc"] - r["R"]["seq_acc"] for r in learned)
        summary.update(seq_A=sA, seq_R=sR, seq_X=sX, verdict=verdict, per_model_drop_A_minus_R=drops)
        print(f"POOLED over {len(learned)} models that learned the task (Path A token accuracy >= {LEARNED_TOKEN_ACC}):")
        print(f"  sequence accuracy  A {sA:.4f}   R {sR:.4f}   X {sX:.4f}")
        print(f"  per-model drop A - R, sorted: {' '.join(f'{x:+.4f}' for x in drops)}")
        print(f"  DECISION: {verdict}")
    else:
        print("no model passed the checks with Path A token accuracy >= 0.99; no decision")
    if not all_ok:
        print("WARNING: some per-model checks failed; those models are excluded from the decision")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"rows": rows, "summary": summary}, f, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
