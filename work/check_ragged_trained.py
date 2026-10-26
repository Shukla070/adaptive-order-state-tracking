"""
Ragged expansion on a TRAINED model: is accuracy unchanged, and what is the
speed with the learned gate itself choosing the factors?

    python work/check_ragged_trained.py --selftest
    python work/check_ragged_trained.py --ckpt checkpoints/gate_B/*.pt
    python work/check_ragged_trained.py --ckpt <files> --out results/ragged_trained.json

Checkpoints come from `exp_gate_train.py --save_dir <dir>`.

WHY
---
`work/check_ragged.py` proved the ragged path equal to Path A on random
weights and timed it with a synthetic copy of the learned allocation. This
closes both gaps: the weights are a trained model's, and the gate deciding
the factors is the trained halting gate, run inside the timed forward pass.

WHAT IT CHECKS (criteria fixed before the first run, 12 October)
-----------------------------------------------------------------
  T0  the checkpoint reloads correctly: per-factor Path A token accuracy on
      the test set is within 0.002 of the accuracy the training run recorded
  T1  for each of {fused Path A, per-factor ragged, fused ragged}:
      token accuracy within 0.001 of per-factor Path A, AND the predicted
      token agrees with Path A's on at least 99.9% of test tokens
  T2  designed to fail: ragged with the output read at the wrong micro-step
      must agree with Path A on fewer than 90% of tokens, or T1 proves nothing
      REVISED 13 Oct after the first run, recorded as a test-design error: the
      mutant only changes tokens that use more than one factor, about 10% of
      them, so its agreement sits near 0.90 whatever happens. The question T2
      asks is whether T1 would catch it, so T2 now requires the mutant to FAIL
      T1's own criteria (agreement < 0.999 or accuracy off by > 0.001).

Added 13 Oct, before re-running, because T1 failed on the untrained model
(seed 11, accuracy 0.42: 90% agreement between ragged and Path A):
  T3  the decisive check, independent of argmax: on 8 test sequences, the
      ragged recurrence output's error against a float64 reference must be no
      more than 2x Path A's own error (+1e-4), for both ragged variants
  M   diagnostic: the top-1 minus top-2 logit margin of Path A at tokens where
      the paths disagree, against all tokens. Near-zero margins there mean the
      disagreements are ties broken differently by bf16 rounding

TIMING (inference, learned gate inside the forward pass)
--------------------------------------------------------
Test sequences are concatenated end to end into batch 32 x length 512 and
32 x 2048 (the task's sequences are 128 long; joining them keeps the real token
mix, so the gate's allocation is the one it learned). Compared with the faster
of the two fixed 4-factor layers (per-factor and fused). The tracked state is
not meaningful across the joins, so these batches are for timing only;
accuracy comes from T0-T2 on the real 128-token test set.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "work"))


def data_csv(data: dict) -> Path:
    """The CSV a training run read, from its saved data settings."""
    group = f"S5_limit_to_het_p{int(round(data['p'] * 1000)):03d}"
    return Path(data["data_dir"]) / f"{group}={data['k']}.csv"


def agreement(a: torch.Tensor, b: torch.Tensor) -> float:
    """Fraction of positions where two prediction tensors agree."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    return (a == b).float().mean().item()


def selftest() -> bool:
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))

    print("check_ragged_trained self-test (CPU)")
    p = data_csv(dict(p=0.1, k=128, data_dir="state_tracking/data"))
    check("S1 data path for p = 0.1 matches the training run's file name",
          p.name == "S5_limit_to_het_p100=128.csv", p.name)
    p2 = data_csv(dict(p=1.0, k=128, data_dir="d"))
    check("S2 data path for p = 1.0", p2.name == "S5_limit_to_het_p1000=128.csv", p2.name)
    a = torch.tensor([[1, 2, 3, 4]])
    check("S3 agreement: identical predictions give 1.0", agreement(a, a) == 1.0)
    check("S4 agreement: one of four differs gives 0.75",
          agreement(a, torch.tensor([[1, 2, 3, 0]])) == 0.75)
    try:
        agreement(a, torch.tensor([1, 2, 3, 4]))
        raised = False
    except ValueError:
        raised = True
    check("S5 agreement refuses tensors of different shapes", raised)
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ckpt", nargs="*", default=[])
    ap.add_argument("--out", default="results/ragged_trained.json")
    ap.add_argument("--no_timing", action="store_true")
    args = ap.parse_args()
    if not selftest():
        sys.exit(1)
    if args.selftest:
        return
    paths = sorted({p for pat in args.ckpt for p in glob.glob(pat)})
    if not paths:
        sys.exit("no checkpoint found; pass --ckpt <files> from exp_gate_train.py --save_dir")

    from exp_matched_compute import build, load, DEV
    from exp_gate_train import set_gate_hard, collect_n_t
    from check_ragged import (set_fused, set_ragged, time_fn, capture,
                              reference_output, rel_err)

    def preds(model, te_x, te_y, fused, ragged, mutant=""):
        set_fused(model, fused)
        set_ragged(model, ragged, mutant)
        out, nts = [], []
        with torch.inference_mode():
            for i in range(0, te_x.shape[0], 128):
                x = te_x[i:i + 128]
                out.append(model(input_ids=x, use_cache=False).logits.argmax(-1))
                nts.append(collect_n_t(model)[0].float().mean().item())
        set_fused(model, False)
        set_ragged(model, False)
        p = torch.cat(out)
        return p, (p == te_y).float().mean().item(), (p == te_y).all(-1).float().mean().item(), \
            sum(nts) / len(nts)

    data_cache = {}
    rows = []
    all_ok = True
    for path in paths:
        ck = torch.load(path, map_location=DEV, weights_only=True)
        b, d, res = ck["build"], ck["data"], ck["result"]
        print("=" * 78)
        print(f"{Path(path).name}   trained: token {res['token_acc']:.4f}, "
              f"cost {res['mean_n_t']:.4f} factors/token, gamma {res['gamma']:g}")
        print("=" * 78)
        if not b["adaptive"]:
            print("  not an adaptive model; skipped")
            continue
        key = (str(data_csv(d)), d["n_train"], d["n_test"])
        if key not in data_cache:
            data_cache[key] = load(data_csv(d), d["n_train"], d["n_test"])
        _, _, te_x, te_y = data_cache[key]

        model = build(**b)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        set_gate_hard(model, True)

        def check(name, cond, detail=""):
            nonlocal_ok[0] &= bool(cond)
            print(f"  [{'PASS' if cond else 'FAIL'}] {name}   {detail}")
        nonlocal_ok = [True]

        pa, acc_a, seq_a, nt_a = preds(model, te_x, te_y, fused=False, ragged=False)
        check("T0 checkpoint reloads: Path A accuracy matches the training run",
              abs(acc_a - res["token_acc"]) <= 0.002,
              f"now {acc_a:.4f}, recorded {res['token_acc']:.4f}; mean order {nt_a:.4f}")
        row = dict(checkpoint=Path(path).name, trained=res,
                   pathA=dict(token_acc=acc_a, seq_acc=seq_a, mean_n_t=nt_a))
        for label, fused, ragged in [("fused Path A", True, False),
                                     ("per-factor ragged", False, True),
                                     ("fused ragged", True, True)]:
            p, acc, seq, nt = preds(model, te_x, te_y, fused, ragged)
            agr = agreement(p, pa)
            check(f"T1 {label}: same accuracy and predictions as Path A",
                  abs(acc - acc_a) <= 0.001 and agr >= 0.999,
                  f"token {acc:.4f} (Path A {acc_a:.4f}), seq {seq:.4f}, "
                  f"agreement {agr:.5f}")
            row[label.replace(" ", "_")] = dict(token_acc=acc, seq_acc=seq,
                                                mean_n_t=nt, agreement=agr)
        p, acc, _, _ = preds(model, te_x, te_y, True, True, mutant="read_at_first")
        agr = agreement(p, pa)
        check("T2 mutant (output read at the wrong step) fails T1's criteria",
              agr < 0.999 or abs(acc - acc_a) > 0.001,
              f"agreement {agr:.4f}, token {acc:.4f} (Path A {acc_a:.4f})")

        # T3: argmax-free check against a float64 reference
        xs = te_x[:8]
        set_fused(model, False)
        ca = capture(model, xs, ragged_on=False)
        err_a = rel_err(ca["o"], reference_output(ca))
        for label, fused in [("per-factor ragged", False), ("fused ragged", True)]:
            set_fused(model, fused)
            cr = capture(model, xs, ragged_on=True)
            err_r = rel_err(cr["o"], reference_output(cr))
            check(f"T3 {label}: as close to the float64 reference as Path A",
                  err_r <= 2 * err_a + 1e-4,
                  f"ragged {err_r:.2e}, Path A {err_a:.2e}")
            row[label.replace(" ", "_")]["ref_err"] = err_r
        set_fused(model, False)
        row["pathA"]["ref_err"] = err_a

        # M: how close to a tie are the tokens where the paths disagree?
        p_r, _, _, _ = preds(model, te_x, te_y, fused=True, ragged=True)
        marg = []
        with torch.inference_mode():
            for i in range(0, te_x.shape[0], 128):
                lg = model(input_ids=te_x[i:i + 128], use_cache=False).logits.float()
                top2 = lg.topk(2, dim=-1).values
                marg.append(top2[..., 0] - top2[..., 1])
        marg = torch.cat(marg)
        dis = p_r != pa
        m_all = marg.median().item()
        m_dis = marg[dis].median().item() if dis.any() else float("nan")
        print(f"  [info] M logit margin (top-1 minus top-2), median: all tokens {m_all:.4f}, "
              f"disagreeing tokens {m_dis:.4f} ({int(dis.sum())} of {dis.numel()})")
        row["margin_median_all"] = m_all
        row["margin_median_disagree"] = m_dis

        if not args.no_timing:
            fixed = build(n_h=4, adaptive=False, n_heads=b["n_heads"], head_dim=b["head_dim"],
                          hidden=b["hidden"], n_layers=b["n_layers"], vocab=b["vocab"],
                          seed=0).eval()
            row["timing"] = []
            print(f"  {'batch x length':<15} {'fix4':>7} {'fix4f':>7} {'A (gate)':>9} "
                  f"{'ragged (gate)':>14} {'order':>6} {'ragged / best4':>15}")
            for B, n_join in [(32, 4), (32, 16)]:
                need = B * n_join
                if te_x.shape[0] < need:
                    print(f"  only {te_x.shape[0]} test sequences; skipping {B} x {n_join * 128}")
                    continue
                x = te_x[:need].reshape(B, n_join * te_x.shape[1])

                def fwd(m):
                    return lambda: m(input_ids=x, use_cache=False)

                set_fused(fixed, False)
                t_f4 = time_fn(fwd(fixed))
                set_fused(fixed, True)
                t_f4f = time_fn(fwd(fixed))
                set_fused(fixed, False)
                set_fused(model, True)
                t_a = time_fn(fwd(model))
                set_ragged(model, True)
                t_r = time_fn(fwd(model))
                with torch.inference_mode():
                    model(input_ids=x, use_cache=False)
                order = collect_n_t(model)[0].float().mean().item()
                set_ragged(model, False)
                set_fused(model, False)
                best4 = min(t_f4, t_f4f)
                print(f"  {B} x {x.shape[1]:<9} {t_f4:7.2f} {t_f4f:7.2f} {t_a:9.2f} "
                      f"{t_r:14.2f} {order:6.3f} {t_r / best4:15.2f}")
                row["timing"].append(dict(batch=B, length=x.shape[1], fixed4_ms=t_f4,
                                          fixed4_fused_ms=t_f4f, pathA_fused_gate_ms=t_a,
                                          ragged_fused_gate_ms=t_r, mean_order=order,
                                          ratio_vs_best4=t_r / best4))
            del fixed
        row["checks_pass"] = nonlocal_ok[0]
        all_ok &= nonlocal_ok[0]
        rows.append(row)
        del model
        torch.cuda.empty_cache()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(rows=rows), indent=2))
    print("\nALL CHECKS PASS" if all_ok else "\nSOME CHECKS FAILED")
    print("Timing: inference, the learned gate runs inside both adaptive forwards;")
    print("joined sequences are for timing only. written:", args.out)


if __name__ == "__main__":
    main()
