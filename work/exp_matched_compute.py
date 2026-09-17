"""
CHECKPOINT A — capability at matched compute.

THE CLAIM
---------
Fixed order is integer-valued, so a fixed-order model can only sit at cost
1, 2, 3 or 4 factors per token. Under heterogeneous demand an adaptive model
sits at E[L] = 1 + 3p and still tracks the group. For every budget B with
E[L] <= B < L_max, no fixed-order model both fits the budget and solves the
task -- and adaptive does. That interval is the operating region fixed order
cannot reach.

THE ARMS
--------
  fixed n_h = 1, 2, 3, 4     cost = n_h
  oracle adaptive            cost = mean n_t = E[L]; n_t = L(g_t), not learned

The oracle arm uses the SAME architecture and the SAME parameter count as
fixed n_h = 4 -- it simply closes gates on tokens that do not need the factors.
So the headline comparison (fixed-4 vs oracle) is matched in parameters and
differs only in compute, which removes the capacity confound that made the S3
smoke test ambiguous.

WHY A STANDALONE TRAINER
------------------------
main.py carries a curriculum, its own metric cadence and early stopping. For a
controlled comparison across five arms, internal consistency matters more than
matching their loop, and one trainer we control removes a class of confounds.
Every arm sees the same data, same seed, same step budget, same optimizer.

    python work/exp_matched_compute.py --p 0.1
    python work/exp_matched_compute.py --p 0.1 --steps 4000 --arms fixed1,fixed4,oracle
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
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oracle_order import (load_ell_lookup, oracle_gates, set_external_gates,
                          effective_cost)                              # noqa: E402

from fla.models import GatedDeltaProductConfig                          # noqa: E402
from fla.models.gated_deltaproduct import GatedDeltaProductForCausalLM  # noqa: E402

DEV = "cuda"
DTYPE = torch.bfloat16
N_SPECIAL = 7          # main.py's tokenizer: group tokens first, then specials
N_GROUP = 120          # |S5|


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load(csv_path: Path, n_train: int, n_test: int):
    """Read the generator's CSV. Token id == element index, so no tokenizer
    is needed -- and no BOS is prepended, which keeps the oracle alignment
    exact (main.py's BOS would shift every position by one)."""
    df = pl.read_csv(csv_path, n_rows=n_train + n_test)
    if df.height < n_train + n_test:
        raise SystemExit(f"{csv_path} has {df.height} rows, need {n_train + n_test}")

    inp = torch.tensor([[int(t) for t in r.split()] for r in df["input"]],
                       dtype=torch.long)
    tgt = torch.tensor([[int(t) for t in r.split()] for r in df["target"]],
                       dtype=torch.long)
    return (inp[:n_train].to(DEV), tgt[:n_train].to(DEV),
            inp[n_train:].to(DEV), tgt[n_train:].to(DEV))


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def reinit_halting_gates(model, init_bias: float, to_float32: bool = True) -> int:
    """Restore the halting gate's initialisation, and return how many were fixed.

    Also casts the gate to float32 when `to_float32`, which is separate from the
    initialisation and equally load-bearing. The surrounding layer runs in
    bfloat16, whose mantissa is 8 bits: a gate weight of magnitude ~1 has a
    resolution of about 0.0078, so an update of order 1e-5 -- the size the
    budget penalty produces once logits are large -- is lost entirely when it is
    added to the weight. Evaluating the sigmoid in float32 (see src/gating.py)
    fixes the vanishing *gradient*; holding the parameters in float32 fixes the
    vanishing *update*. Both are needed, and neither is sufficient alone.

    The gate is a small MLP beside a Triton kernel, so the cost is negligible.
    HaltingGate.forward casts its input to the gate's dtype and its output back
    to the input's, so the rest of the layer is unaffected.

    MUST be called after the HuggingFace model is constructed. `PreTrainedModel`
    runs `post_init()` -> `_init_weights` over every submodule once __init__ has
    finished, and that hook matches `nn.Linear`:

        nn.init.normal_(module.weight, std=initializer_range)
        nn.init.zeros_(module.bias)

    The gate's output layer is an nn.Linear, so both properties src/gating.py
    sets deliberately are destroyed: the zero weight that makes the gate
    input-independent at step 0, and the positive bias that makes it start open.
    A gate left at bias 0 begins at n_t = 0.94 of 4 factors instead of 3.82, so
    the model does not start as fixed-order DeltaProduct(K).

    work/check_gate_init.py is the regression test for this and fails against a
    build() that omits this call.
    """
    import torch.nn as nn
    n = 0
    for mod in model.modules():
        if not getattr(mod, "adaptive_order", False):
            continue
        g = getattr(mod, "halting_gate", None)
        if g is None:
            continue
        if to_float32:
            g.float()
        with torch.no_grad():
            nn.init.zeros_(g.mlp[-1].weight)
            nn.init.constant_(g.mlp[-1].bias, init_bias)
        n += 1
    if n == 0:
        raise RuntimeError(
            "adaptive_order=True but no halting gate was found to initialise.")
    return n


def build(n_h: int, adaptive: bool, n_heads: int, head_dim: int, hidden: int,
          n_layers: int, vocab: int, seed: int, gate_init_bias: float = 4.0):
    """`gate_init_bias` is the halting gate's output bias at init. The default
    4.0 is the value every experiment before 15 Sept ran with, so existing
    callers are unaffected. It is exposed because sigmoid(4.0) = 0.982 has
    derivative 0.018, and a gate that starts that far into saturation may not
    become input-dependent at all (see exp_gate_train.py)."""
    torch.manual_seed(seed)
    cfg = GatedDeltaProductConfig(
        hidden_size=hidden,
        num_hidden_layers=n_layers,
        num_heads=n_heads,
        head_dim=head_dim,
        expand_v=1,
        vocab_size=vocab,
        allow_neg_eigval=True,      # required: reflections need eigenvalue -1
        num_householder=n_h,
        fuse_cross_entropy=False,
        max_position_embeddings=2048,
    )
    cfg.adaptive_order = adaptive
    cfg.gate_hard = True            # integer orders, so cost is what we report
    cfg.gate_init_open_bias = gate_init_bias
    m = GatedDeltaProductForCausalLM(cfg).to(DEV, DTYPE)
    if adaptive:
        # AFTER the cast: .to(DTYPE) would otherwise undo the gate's float32.
        reinit_halting_gates(m, gate_init_bias)
    if adaptive:
        n_adaptive = sum(1 for mod in m.modules()
                         if getattr(mod, "adaptive_order", False))
        if n_adaptive == 0:
            raise SystemExit(
                "adaptive_order did not reach the layer. Is the patch to "
                "fla/models/gated_deltaproduct/modeling_gated_deltaproduct.py "
                "applied?")
    return m


@torch.no_grad()
def evaluate(model, inp, tgt, lut, max_order, use_oracle, batch=128):
    model.eval()
    tok_correct = tok_total = 0
    seq_correct = 0
    for i in range(0, inp.shape[0], batch):
        x, y = inp[i:i + batch], tgt[i:i + batch]
        if use_oracle:
            set_external_gates(model, oracle_gates(x, lut, max_order, dtype=DTYPE))
        logits = model(input_ids=x).logits
        pred = logits.float().argmax(-1)
        eq = pred == y
        tok_correct += eq.sum().item()
        tok_total += eq.numel()
        seq_correct += eq.all(-1).sum().item()
    if use_oracle:
        set_external_gates(model, None)
    model.train()
    return tok_correct / tok_total, seq_correct / inp.shape[0]


def run_arm(name, n_h, use_oracle, data, lut, args):
    tr_x, tr_y, te_x, te_y = data
    model = build(n_h, use_oracle, args.n_heads, args.head_dim, args.hidden,
                  args.n_layers, args.vocab, args.seed)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    t0 = time.perf_counter()
    g = torch.Generator(device=DEV).manual_seed(args.seed)
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, tr_x.shape[0], (args.batch,), device=DEV, generator=g)
        x, y = tr_x[idx], tr_y[idx]
        if use_oracle:
            set_external_gates(model, oracle_gates(x, lut, n_h, dtype=DTYPE))
        logits = model(input_ids=x).logits
        loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % args.log_every == 0 or step == 1:
            print(f"    {name:<10} step {step:>5}  loss {loss.item():.4f}")
    dt = time.perf_counter() - t0
    if use_oracle:
        set_external_gates(model, None)

    tok, seq = evaluate(model, te_x, te_y, lut, n_h, use_oracle)
    cost = (effective_cost(tr_x[:2000], lut, n_h) if use_oracle else float(n_h))
    return dict(arm=name, n_h=n_h, oracle=use_oracle, cost=cost,
                token_acc=tok, seq_acc=seq, params=n_par, train_s=dt)


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
    ap.add_argument("--log_every", type=int, default=1000)
    ap.add_argument("--max_order", type=int, default=4)
    ap.add_argument("--arms", default="fixed1,fixed2,fixed3,fixed4,oracle")
    ap.add_argument("--out", default="results/matched_compute.json")
    args = ap.parse_args()
    args.vocab = N_GROUP + N_SPECIAL

    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA")

    group = f"S5_limit_to_het_p{int(round(args.p * 1000)):03d}"
    csv = Path(args.data_dir) / f"{group}={args.k}.csv"
    meta = Path(args.data_dir) / f"{group}_meta.json"
    for f in (csv, meta):
        if not f.exists():
            raise SystemExit(f"missing {f}\nGenerate it:\n"
                             f"  python work/gen_heterogeneous.py --p {args.p} "
                             f"--k {args.k} --samples {args.n_train + args.n_test}")

    m = json.loads(meta.read_text())
    lut = load_ell_lookup(meta, args.vocab, device=DEV)
    data = load(csv, args.n_train, args.n_test)

    print("=" * 78)
    print(f"CHECKPOINT A — capability at matched compute   (p = {args.p})")
    print("=" * 78)
    print(f"  data      {csv.name}   {args.n_train} train / {args.n_test} test, k={args.k}")
    print(f"  demand    E[L] = {m['empirical_mean_ell']:.3f}   "
          f"fixed order must provision L_max = {m['fixed_order_cost']}")
    print(f"  budget    {args.steps} steps x batch {args.batch}, lr {args.lr} "
          f"(constant), seed {args.seed}")
    print(f"  model     {args.n_layers} layer, {args.n_heads} heads x {args.head_dim}\n")

    specs = {f"fixed{i}": (i, False) for i in (1, 2, 3, 4)}
    specs["oracle"] = (args.max_order, True)

    rows = []
    for name in args.arms.split(","):
        name = name.strip()
        if name not in specs:
            raise SystemExit(f"unknown arm {name}; choose from {list(specs)}")
        n_h, use_oracle = specs[name]
        print(f"  -- {name} (n_h={n_h}{', oracle gates' if use_oracle else ''})")
        rows.append(run_arm(name, n_h, use_oracle, data, lut, args))
        print()

    print("=" * 78)
    print(f"{'arm':<10} {'cost/token':>11} {'token acc':>10} {'seq acc':>9} "
          f"{'params':>9} {'train s':>8}")
    print("-" * 78)
    for r in rows:
        print(f"{r['arm']:<10} {r['cost']:>11.3f} {r['token_acc']:>10.4f} "
              f"{r['seq_acc']:>9.4f} {r['params']/1e3:>8.0f}k {r['train_s']:>8.0f}")

    # the claim, evaluated
    SOLVED = 0.99
    solved = [r for r in rows if r["token_acc"] >= SOLVED]
    oracle = next((r for r in rows if r["oracle"]), None)
    fixed_solved = [r for r in solved if not r["oracle"]]
    print("=" * 78)
    if oracle and oracle["token_acc"] >= SOLVED and fixed_solved:
        cheapest_fixed = min(r["cost"] for r in fixed_solved)
        if oracle["cost"] < cheapest_fixed:
            print(f"CLAIM HOLDS at p={args.p}")
            print(f"  oracle solves the task at cost {oracle['cost']:.3f}/token")
            print(f"  cheapest fixed order that solves it costs {cheapest_fixed:.0f}")
            print(f"  no fixed-order model both fits a budget in "
                  f"[{oracle['cost']:.3f}, {cheapest_fixed:.0f}) and solves the task")
        else:
            print(f"CLAIM DOES NOT HOLD: oracle cost {oracle['cost']:.3f} is not "
                  f"below the cheapest solving fixed order ({cheapest_fixed:.0f})")
    elif oracle and oracle["token_acc"] < SOLVED:
        print("INCONCLUSIVE: the oracle did not solve the task. Either the step "
              "budget is too small, or L is not the right per-token demand "
              "(recall A5 trains at n_h=2 despite theory requiring 4).")
    else:
        print("INCONCLUSIVE: no fixed-order arm solved the task. Increase --steps.")
    print("=" * 78)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"p": args.p, "k": args.k, "config": vars(args), "meta": m, "results": rows},
        indent=2, default=str))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
