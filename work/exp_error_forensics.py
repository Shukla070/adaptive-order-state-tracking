"""
ERROR FORENSICS -- why does fixed n_h=3 score 0.54 sequence accuracy on a task
whose hard tokens need 4 Householder factors?

THE PUZZLE, STATED IN NUMBERS
-----------------------------
From the matched-compute run at p=0.1:

    fixed3    token acc 0.9477    seq acc 0.5425

Those two numbers are hard to hold together. Take the two obvious failure models
and neither fits:

  independent per-position errors at rate e:
      e = 1 - 0.9477 = 0.0523  =>  seq acc = 0.9477^128 = 0.001
      observed 0.5425.  Off by 500x.

  one bad hard token poisons the rest of the sequence:
      seq acc 0.5425 over Binom(128, 0.1) hard tokens implies a per-hard-token
      success of q = 0.952, which predicts token acc 0.746.
      observed 0.9477.  Off by 20 points.

So errors are neither independent nor permanent. They are CLUSTERED: roughly
54% of sequences are perfect, and the other 46% still get ~89% of their tokens
right. Something structured is happening, and E1-E5 of the reachability file
proved it is not a cleverer factorisation of the 5-cycle -- three factors leave
68% relative error, which cannot survive a 128-step running product.

WHAT THIS FILE MEASURES
-----------------------
  F1  accuracy vs the number of hard (L=4) tokens seen so far.
      A clean q^h curve means each hard token is an independent coin flip.
      A flat curve means hard tokens are not the failure mode at all.

  F2  error persistence. Given a wrong prediction at t, how often is t+1 wrong?
      ~1.0 means the recurrent state is corrupted and stays corrupted.
      ~base rate means the state is fine and only the readout slipped.

  F3  recovery length. If errors do NOT persist, how many steps until correct
      again? A finite recovery length in a group-tracking task is strange --
      the state has no reason to self-correct -- and would point at the additive
      v k^T channel rather than the transition matrices.

  F4  accuracy vs L of the CUMULATIVE target, not the input token. If the model
      is systematically worse on targets that are themselves 5-cycles, it is
      failing at the readout, not the recurrence.

  F5  the position of the first error, against the position of the first hard
      token. If they coincide, hard tokens are the trigger. If the first error
      comes long before the first hard token, they are not.

Run fixed1 / fixed3 / fixed4 so the pathology can be read against a model that
cannot do the task and one that can.

    python work/exp_error_forensics.py --p 0.1
    python work/exp_error_forensics.py --p 0.1 --arms fixed3 --steps 20000
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

# fla / CUDA are imported inside main() on purpose: F1-F5 below are pure tensor
# code, and keeping them importable without a GPU is what lets _tests() run them
# against synthetic data where the right answer is known in advance.
DEV = "cuda"
DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------

def train(model, tr_x, tr_y, steps, batch, lr, seed, log_every, name,
          set_external_gates=None, oracle_gates=None,
          lut=None, n_h=None, use_oracle=False):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    g = torch.Generator(device=DEV).manual_seed(seed)
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        idx = torch.randint(0, tr_x.shape[0], (batch,), device=DEV, generator=g)
        x, y = tr_x[idx], tr_y[idx]
        if use_oracle:
            set_external_gates(model, oracle_gates(x, lut, n_h, dtype=DTYPE))
        loss = F.cross_entropy(
            model(input_ids=x).logits.float().flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % log_every == 0 or step == 1:
            print(f"    {name:<8} step {step:>6}  loss {loss.item():.4f}")
    if use_oracle:
        set_external_gates(model, None)
    return time.perf_counter() - t0


@torch.no_grad()
def collect(model, inp, tgt, batch=128):
    """Per-position correctness, as a bool tensor shaped like the targets."""
    model.eval()
    out = []
    for i in range(0, inp.shape[0], batch):
        pred = model(input_ids=inp[i:i + batch]).logits.float().argmax(-1)
        out.append(pred == tgt[i:i + batch])
    model.train()
    return torch.cat(out)


# ---------------------------------------------------------------------------
# the five forensics
# ---------------------------------------------------------------------------

def F1(correct, hard_cum, max_h=12):
    """Accuracy at position t, bucketed by how many hard tokens preceded t."""
    print("  F1  accuracy vs number of hard (L=4) tokens seen so far")
    print(f"      {'h':>4} {'positions':>11} {'accuracy':>10} {'q^h fit':>10}")
    rows = []
    acc0 = None
    for h in range(0, max_h + 1):
        m = hard_cum == h
        n = int(m.sum())
        if n < 200:
            continue
        a = correct[m].float().mean().item()
        if acc0 is None:
            acc0 = a
        rows.append(dict(h=h, n=n, acc=a))
    if len(rows) >= 2:
        # q estimated from the last bucket, so the fit column is a prediction
        # for the middle of the range rather than a curve fitted to it
        last = rows[-1]
        q = (last["acc"] / acc0) ** (1.0 / last["h"]) if last["h"] else 1.0
    else:
        q = 1.0
    for r in rows:
        r["fit"] = acc0 * q ** r["h"]
        print(f"      {r['h']:>4} {r['n']:>11} {r['acc']:>10.4f} {r['fit']:>10.4f}")
    print(f"      implied per-hard-token success q = {q:.4f}")
    return {"buckets": rows, "q": q}


def F2(correct):
    """P(wrong at t+1 | wrong at t) against the base rate."""
    prev, nxt = correct[:, :-1], correct[:, 1:]
    base = 1.0 - nxt.float().mean().item()
    m = ~prev
    persist = (1.0 - nxt[m].float().mean().item()) if m.any() else float("nan")
    m2 = prev
    after_ok = (1.0 - nxt[m2].float().mean().item()) if m2.any() else float("nan")
    print("  F2  error persistence")
    print(f"      P(wrong at t+1)                    = {base:.4f}   (base rate)")
    print(f"      P(wrong at t+1 | wrong at t)       = {persist:.4f}")
    print(f"      P(wrong at t+1 | correct at t)     = {after_ok:.4f}")
    # NOT persist/base. When errors are absorbing the base rate is itself made
    # large by the persistence, so that ratio collapses toward 1 exactly when
    # persistence is total -- the self-test caught this scoring an absorbing
    # world at 3.7x. Comparing against the post-correct rate has no such
    # feedback: it is the odds an error CONTINUES over the odds one STARTS.
    ratio = persist / after_ok if after_ok > 0 else float("inf")
    print(f"      continue / start ratio             = {ratio:.1f}x")
    if ratio > 10:
        print("      -> errors persist: the recurrent state is corrupted and")
        print("         the model has no way back.")
    elif ratio < 3:
        print("      -> errors do NOT persist: the state survives and the")
        print("         readout slipped on that token alone.")
    else:
        print("      -> partial persistence.")
    return {"base": base, "persist": persist, "after_ok": after_ok, "ratio": ratio}


def F3(correct, max_run=12):
    """How long a wrong run lasts, once it starts."""
    c = correct.cpu()
    runs = []
    for row in c:
        i, T = 0, row.shape[0]
        while i < T:
            if not row[i]:
                j = i
                while j < T and not row[j]:
                    j += 1
                runs.append(j - i)
                i = j
            else:
                i += 1
    print("  F3  length of wrong-runs")
    if not runs:
        print("      no errors at all")
        return {"runs": 0}
    t = torch.tensor(runs, dtype=torch.float64)
    hist = {}
    for L in range(1, max_run + 1):
        hist[L] = int((t == L).sum())
    tail = int((t > max_run).sum())
    print(f"      {len(runs)} wrong-runs, mean length {t.mean():.2f}, "
          f"median {t.median():.0f}, max {int(t.max())}")
    line = "      " + "  ".join(f"{L}:{hist[L]}" for L in range(1, 7))
    print(line + f"   >{max_run}:{tail}")
    if t.mean() < 3:
        print("      -> short bursts. The model RECOVERS, which a corrupted")
        print("         group-tracking state has no mechanism to do.")
    return {"n_runs": len(runs), "mean": t.mean().item(),
            "median": t.median().item(), "max": int(t.max()),
            "hist": hist, "tail_gt_max": tail}


def F4(correct, tgt, ell_lut):
    """Accuracy bucketed by L of the cumulative target element."""
    lt = ell_lut[tgt]
    print("  F4  accuracy vs L of the cumulative TARGET")
    print(f"      {'L':>4} {'positions':>11} {'accuracy':>10}")
    rows = []
    for L in range(0, 5):
        m = lt == L
        n = int(m.sum())
        if n < 200:
            continue
        a = correct[m].float().mean().item()
        rows.append(dict(ell=L, n=n, acc=a))
        print(f"      {L:>4} {n:>11} {a:>10.4f}")
    if rows:
        spread = max(r["acc"] for r in rows) - min(r["acc"] for r in rows)
        print(f"      spread {spread:.4f}")
        if spread > 0.15:
            print("      -> strongly target-dependent: a readout/decoding failure.")
        else:
            print("      -> roughly flat: not a decoding failure.")
    return {"buckets": rows}


def F5(correct, hard_mask):
    """First error position vs first hard token position."""
    T = correct.shape[1]
    big = T + 1
    first_err = torch.where(~correct, torch.arange(T, device=correct.device)
                            .expand_as(correct), torch.full_like(correct, big,
                                                                 dtype=torch.long))
    first_err = first_err.min(dim=1).values
    first_hard = torch.where(hard_mask, torch.arange(T, device=hard_mask.device)
                             .expand_as(hard_mask),
                             torch.full_like(hard_mask, big, dtype=torch.long))
    first_hard = first_hard.min(dim=1).values

    has_err = first_err <= T
    print("  F5  first error vs first hard token")
    print(f"      sequences with any error : {int(has_err.sum())} / {correct.shape[0]}")
    if not bool(has_err.any()):
        return {"n_err_seqs": 0}
    fe, fh = first_err[has_err].float(), first_hard[has_err].float()
    delta = fe - fh
    before = int((delta < 0).sum())
    at = int((delta == 0).sum())
    print(f"      mean first-error position      = {fe.mean():.1f}")
    print(f"      mean first-hard-token position = {fh.mean():.1f}")
    print(f"      first error BEFORE first hard token : {before} "
          f"({100*before/len(fe):.1f}%)")
    print(f"      first error AT the first hard token : {at} "
          f"({100*at/len(fe):.1f}%)")
    if before > 0.3 * len(fe):
        print("      -> a large share of failures start before any hard token.")
        print("         Hard tokens are not the whole story.")
    return {"n_err_seqs": int(has_err.sum()),
            "mean_first_err": fe.mean().item(),
            "mean_first_hard": fh.mean().item(),
            "frac_before": before / len(fe), "frac_at": at / len(fe)}


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, default=0.1)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--data_dir", default="state_tracking/data")
    ap.add_argument("--n_train", type=int, default=100000)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_heads", type=int, default=12)
    ap.add_argument("--head_dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n_layers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=666)
    ap.add_argument("--log_every", type=int, default=2000)
    ap.add_argument("--arms", default="fixed1,fixed3,fixed4")
    ap.add_argument("--out", default="results/error_forensics.json")
    args = ap.parse_args()

    from exp_matched_compute import build, load, N_GROUP, N_SPECIAL
    from oracle_order import load_ell_lookup
    args.vocab = N_GROUP + N_SPECIAL

    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA")

    group = f"S5_limit_to_het_p{int(round(args.p * 1000)):03d}"
    csv = Path(args.data_dir) / f"{group}={args.k}.csv"
    meta = Path(args.data_dir) / f"{group}_meta.json"
    for f in (csv, meta):
        if not f.exists():
            raise SystemExit(f"missing {f}; run gen_heterogeneous.py --p {args.p}")

    lut = load_ell_lookup(meta, args.vocab, device=DEV)
    tr_x, tr_y, te_x, te_y = load(csv, args.n_train, args.n_test)

    hard_mask = lut[te_x] == 4                      # (B, T) bool
    hard_cum = hard_mask.long().cumsum(dim=1)       # hard tokens seen INCLUDING t

    print("=" * 78)
    print(f"ERROR FORENSICS   p={args.p}  k={args.k}  steps={args.steps}")
    print("=" * 78)
    print(f"  test set {te_x.shape[0]} x {te_x.shape[1]}, "
          f"{float(hard_mask.float().mean()):.3f} of tokens are hard (L=4)")
    print()

    specs = {"fixed1": 1, "fixed2": 2, "fixed3": 3, "fixed4": 4}
    out = {}
    for name in [a.strip() for a in args.arms.split(",")]:
        n_h = specs[name]
        print("-" * 78)
        print(f"ARM {name}  (n_h={n_h})")
        print("-" * 78)
        model = build(n_h, False, args.n_heads, args.head_dim, args.hidden,
                      args.n_layers, args.vocab, args.seed)
        dt = train(model, tr_x, tr_y, args.steps, args.batch, args.lr,
                   args.seed, args.log_every, name)
        correct = collect(model, te_x, te_y)
        tok = correct.float().mean().item()
        seq = correct.all(-1).float().mean().item()
        print(f"    token acc {tok:.4f}   seq acc {seq:.4f}   ({dt:.0f}s)")
        print()

        # the two naive models this is meant to discriminate between
        indep = tok ** correct.shape[1]
        print(f"  sanity: if errors were independent per position, seq acc would")
        print(f"          be {tok:.4f}^{correct.shape[1]} = {indep:.4g}; observed {seq:.4f}")
        print()

        res = {"token_acc": tok, "seq_acc": seq, "train_s": dt,
               "indep_prediction": indep}
        res["F1"] = F1(correct, hard_cum)
        print()
        res["F2"] = F2(correct)
        print()
        res["F3"] = F3(correct)
        print()
        res["F4"] = F4(correct, te_y, lut)
        print()
        res["F5"] = F5(correct, hard_mask)
        print()
        out[name] = res
        del model
        torch.cuda.empty_cache()

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"config": vars(args), "results": out},
                            indent=2, default=str))
    print("=" * 78)
    print(f"written: {p}")
    print("=" * 78)


def _tests() -> bool:
    """Run F1-F5 against synthetic data whose failure mode is known.

    A diagnostic that cannot tell the two hypotheses apart on data built to
    contain exactly one of them will not tell them apart on the real model
    either. So: build an independent-error world and an absorbing-error world,
    and require the diagnostics to separate them.
    """
    global DEV
    DEV = "cpu"
    torch.manual_seed(0)
    B, T, P_HARD = 3000, 128, 0.10
    hard_mask = torch.rand(B, T) < P_HARD
    hard_cum = hard_mask.long().cumsum(dim=1)

    # world A: errors independent per position, nothing to do with hard tokens
    A = torch.rand(B, T) > 0.05

    # world B: a hard token breaks the state with prob 0.05, and it never heals
    broke = hard_mask & (torch.rand(B, T) < 0.05)
    Bc = ~(broke.cumsum(dim=1) > 0)

    ok = True
    print("=" * 78)
    print("SELF-TEST -- can the forensics separate independent from absorbing?")
    print("=" * 78)
    for name, c, expect_persist in (("independent", A, False),
                                    ("absorbing", Bc, True)):
        print(f"\n--- synthetic world: {name} "
              f"(token {c.float().mean():.4f}, seq {c.all(-1).float().mean():.4f})")
        f1 = F1(c, hard_cum)
        print()
        f2 = F2(c)
        print()
        f3 = F3(c)
        print()
        f5 = F5(c, hard_mask)

        if expect_persist:
            t_ratio = f2["ratio"] > 10
            t_run = f3["mean"] > 10
            t_q = f1["q"] < 0.99
        else:
            t_ratio = f2["ratio"] < 3
            t_run = f3["mean"] < 3
            t_q = f1["q"] > 0.97
        print(f"\n    persistence separates : {t_ratio}")
        print(f"    run length separates   : {t_run}")
        print(f"    q^h curve separates    : {t_q}")
        ok &= t_ratio and t_run and t_q

    print("\n" + "=" * 78)
    print("SELF-TEST PASSED" if ok else "SELF-TEST FAILED")
    print("=" * 78)
    return ok


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(0 if _tests() else 1)
    main()
