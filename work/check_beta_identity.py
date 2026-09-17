"""
BLOCKING CHECK (Week 1, Day 3): does beta = 0 make a Householder factor exactly
the identity in the REAL fla implementation?

The maths is already proven (reference_deltaproduct.py, T1: bit-exact). This
checks the actual Triton path.

NOTE ON PRECISION: chunk_gated_delta_rule asserts bfloat16 -- float32 is not
supported. bf16 carries ~3 decimal digits, and the two paths being compared run
sequences of different length (4T vs T), so their chunk boundaries and therefore
their accumulation order differ. Bit-exact agreement is NOT expected and would
not be meaningful. So this is a signal-vs-noise test:

    gated-off  |o(n_h=4, betas 1..3 = 0) - o(n_h=1)|     <- should be ~bf16 noise
    ungated    |o(n_h=4, all betas live)  - o(n_h=1)|     <- should be much larger

PASS iff the gated-off difference is small in absolute terms AND at least 10x
smaller than the ungated difference.

Run from the repo root:
    python work/check_beta_identity.py
"""

import torch
import torch.nn as nn
from fla.layers.gated_deltaproduct import GatedDeltaProduct

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16          # required by chunk_gated_delta_rule


class ZeroBeta(nn.Module):
    """Stand-in for b_projs[i] driving sigmoid() to exactly 0."""

    def __init__(self, out_features):
        super().__init__()
        self.out_features = out_features

    def forward(self, x):
        return torch.full((*x.shape[:-1], self.out_features),
                          float("-inf"), device=x.device, dtype=x.dtype)


def build(num_householder, cfg, seed=0):
    torch.manual_seed(seed)
    return GatedDeltaProduct(num_householder=num_householder, **cfg).to(DEV, DTYPE)


def reldiff(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()


def main():
    print(f"device = {DEV} | torch = {torch.__version__} | dtype = {DTYPE}")
    if DEV == "cpu":
        print("WARNING: no CUDA — fla chunk kernels are GPU-only, test is meaningless here.")
        return 1

    cfg = dict(hidden_size=128, head_dim=32, num_heads=4, expand_v=1,
               use_gate=True, use_forget_gate=True, use_short_conv=True,
               allow_neg_eigval=True, layer_idx=0)
    B, T, K = 2, 128, 4

    m4_gated = build(K, cfg)
    m4_live  = build(K, cfg)          # same seed -> same weights, gates left ON
    m1       = build(1, cfg)

    sd4, sd1 = m4_gated.state_dict(), m1.state_dict()
    shared = {k: sd4[k] for k in sd1 if k in sd4 and sd4[k].shape == sd1[k].shape}
    m1.load_state_dict(shared, strict=False)
    print(f"shared {len(shared)}/{len(sd1)} tensors into the n_h=1 layer")

    n_heads_out = m4_gated.b_projs[0].out_features
    for i in range(1, K):
        m4_gated.b_projs[i] = ZeroBeta(n_heads_out).to(DEV, DTYPE)

    torch.manual_seed(123)
    x = torch.randn(B, T, cfg["hidden_size"], device=DEV, dtype=DTYPE)

    for m in (m4_gated, m4_live, m1):
        m.eval()
    with torch.no_grad():
        o_gated = m4_gated(x)[0]
        o_live  = m4_live(x)[0]
        o_one   = m1(x)[0]

    r_gated = reldiff(o_gated, o_one)
    r_live  = reldiff(o_live,  o_one)

    print(f"\nshapes: gated {tuple(o_gated.shape)} | n_h=1 {tuple(o_one.shape)}")
    print(f"  SIGNAL  betas 1..3 = 0   vs n_h=1 : max rel diff = {r_gated:.3e}")
    print(f"  CONTROL all betas live   vs n_h=1 : max rel diff = {r_live:.3e}")
    print(f"  ratio (control / signal)          : {r_live / max(r_gated, 1e-12):.1f}x")
    print(f"  NaN or Inf in gated output        : "
          f"{bool(torch.isnan(o_gated).any() or torch.isinf(o_gated).any())}")

    ok = (r_gated < 5e-2) and (r_live > 10 * r_gated) and not torch.isnan(o_gated).any()
    print(f"\n{'PASS' if ok else 'FAIL'} — beta=0 "
          f"{'acts as the identity factor in fla (difference is bf16 noise)' if ok else 'does NOT act as identity; investigate'}")
    if not ok:
        print("\nIf FAIL, try in order:")
        print("  1. use_short_conv=False   (isolates the conv path)")
        print("  2. use_forget_gate=False  (uses chunk_delta_rule instead; checks")
        print("     whether decay g is still applied on gated-off micro-steps)")
        print("  3. print beta right after the sigmoid in gated_deltaproduct.py:253")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
