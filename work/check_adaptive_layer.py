"""
Verify the adaptive-order patch to fla/layers/gated_deltaproduct.py.

Three properties, in order of importance:

  A. adaptive_order=False must be IDENTICAL to upstream. Every baseline in
     Phase 1 runs through this file; if the patch perturbs the default path,
     every number we produce is suspect.
  B. adaptive_order=True with gates open must match the fixed-order model to
     within the gate's initialisation slack.
  C. n_t must be exposed so the training loop can add the budget penalty.

Run from the repo root (needs a GPU — the kernels are Triton/bf16):
    python work/check_adaptive_layer.py
"""

import torch
from fla.layers.gated_deltaproduct import GatedDeltaProduct

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

CFG = dict(hidden_size=128, head_dim=32, num_heads=4, expand_v=1,
           use_gate=True, use_forget_gate=True, use_short_conv=True,
           allow_neg_eigval=True, layer_idx=0)
B, T, K = 2, 128, 4


def build(seed=0, **extra):
    torch.manual_seed(seed)
    return GatedDeltaProduct(num_householder=K, **CFG, **extra).to(DEV, DTYPE)


def reldiff(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()


def main():
    if DEV == "cpu":
        raise SystemExit("needs CUDA — fla chunk kernels are GPU only")
    print(f"device={DEV} dtype={DTYPE}\n")

    torch.manual_seed(123)
    x = torch.randn(B, T, CFG["hidden_size"], device=DEV, dtype=DTYPE)

    # --- A. default path unchanged ----------------------------------------
    plain = build(seed=0)
    also_plain = build(seed=0, adaptive_order=False)
    with torch.no_grad():
        o1, o2 = plain(x)[0], also_plain(x)[0]
    a_err = (o1 - o2).abs().max().item()
    a_ok = a_err == 0.0 and plain.halting_gate is None
    print(f"A  adaptive_order=False is upstream : max|do| = {a_err:.2e}, "
          f"gate is None = {plain.halting_gate is None}   "
          f"[{'PASS' if a_ok else 'FAIL'}]")

    # --- B. open gates ~ fixed order --------------------------------------
    adaptive = build(seed=0, adaptive_order=True)
    # share every weight the two models have in common
    sd = plain.state_dict()
    shared = {k: v for k, v in sd.items() if k in adaptive.state_dict()}
    adaptive.load_state_dict(shared, strict=False)

    with torch.no_grad():
        adaptive.halting_gate.mlp[-1].bias.fill_(20.0)      # sigmoid(20)=1 in bf16
        o3 = adaptive(x)[0]
    b_err = reldiff(o3, o1)
    n_t = adaptive.last_n_t
    b_ok = b_err < 5e-2 and abs(n_t.mean().item() - K) < 1e-3
    print(f"B  open gates ~ fixed order        : rel diff = {b_err:.2e}, "
          f"mean n_t = {n_t.mean().item():.4f} (K={K})   "
          f"[{'PASS' if b_ok else 'FAIL'}]")

    # --- C. closed gates collapse the order -------------------------------
    with torch.no_grad():
        adaptive.halting_gate.mlp[-1].bias.copy_(
            torch.tensor([20.0, -20.0, -20.0, -20.0], device=DEV, dtype=DTYPE))
        o4 = adaptive(x)[0]
    n_t1 = adaptive.last_n_t
    order1 = build(seed=0)
    order1_sd = {k: v for k, v in plain.state_dict().items()
                 if k in order1.state_dict()}
    order1.load_state_dict(order1_sd, strict=False)
    c_ok = abs(n_t1.mean().item() - 1.0) < 1e-2 and not torch.isnan(o4).any()
    print(f"C  gates closed to order 1         : mean n_t = {n_t1.mean().item():.4f}, "
          f"NaN = {bool(torch.isnan(o4).any())}   [{'PASS' if c_ok else 'FAIL'}]")

    # --- D. gradients reach the gate --------------------------------------
    # NB: rebuild. Check C left the bias at +-20, where the sigmoid is fully
    # saturated and gradients are ~1e-9 -- testing D on that gate would pass
    # vacuously while telling us nothing.
    adaptive = build(seed=0, adaptive_order=True)
    adaptive.load_state_dict(shared, strict=False)
    adaptive.zero_grad()
    out = adaptive(x)[0]
    (out.float().pow(2).mean() + 0.1 * adaptive.last_n_t.float().mean()).backward()
    gnorm = sum(p.grad.abs().sum().item()
                for p in adaptive.halting_gate.parameters() if p.grad is not None)
    # A saturated gate cannot learn. At init_open_bias=4.0 the sigmoid slope is
    # ~0.018, so gradients are small but usable; demand something above noise.
    d_ok = gnorm > 1e-4
    print(f"D  gradients reach the gate        : |grad| = {gnorm:.6f}   "
          f"[{'PASS' if d_ok else 'FAIL'}]")
    if not d_ok:
        print("    -> gate is saturated; lower gate_init_open_bias.")

    ok = a_ok and b_ok and c_ok and d_ok
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    if not a_ok:
        print("A failing means the patch changed the default path. Revert with:")
        print("  cp flash-linear-attention/fla/layers/gated_deltaproduct.py.orig \\")
        print("     flash-linear-attention/fla/layers/gated_deltaproduct.py")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
