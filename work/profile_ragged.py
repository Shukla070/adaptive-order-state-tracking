"""
Where does ragged expansion spend its time?

    python work/profile_ragged.py --selftest   # categoriser known-answer test, CPU, instant
    python work/profile_ragged.py              # GPU profile, ~2 min
    python work/profile_ragged.py --out results/ragged_profile.json

WHY THIS EXISTS
---------------
`work/check_ragged.py` (version 2 packing) measured ragged expansion at 0.65-0.71
of the fixed-4 time on the three large cells, where the arithmetic from fixed
orders 1-4 suggests ~0.4. With every factor kept, ragged took ~12-14 ms longer
than Path A at batch 32, length 2048. This script splits each forward pass into
GPU kernel time by category, so the next change targets the measured cost
instead of a guess.

WHAT IT REPORTS
---------------
For batch x length in {32 x 2048, 8 x 2048}, four configurations:
  fixed4         a layer built with 4 factors (no gate)
  pathA_s5mix    adaptive layer, all factors computed, gates at the learned
                 allocation (order 4 for 10% of tokens, else 1)
  ragged_s5mix   the same with ragged expansion
  ragged_all4    ragged expansion with every factor kept (the packing overhead)
For each: wall time per forward (cuda events), total GPU kernel time, kernel
launches, and kernel time per category:
  recurrence  the delta-rule kernels (chunked WY form, l2norm, outputs)
  packing     index_copy / index_select / where / cumsum / scatter / gather
  matmul      projections (GEMMs)
  conv        short convolutions
  norm_act    norms and activations
  copy_elem   copies, fills, cat/stack, generic elementwise
  other       anything unmatched (listed by name, so nothing is hidden)
Wall time minus kernel time is time the GPU sat idle (CPU launch overhead or
host synchronisation).

Weights are random and the allocation is synthetic: this measures cost, not
accuracy.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

CATEGORIES = [
    ("recurrence", r"delta|wy_repr|solve_tril|chunk_fwd|chunk_.*kernel|fwd_kernel_[ho]\b|"
                   r"l2norm|merge_16x16|kkt|chunk_local_cumsum|recompute_w_u"),
    ("packing", r"index_copy|indexcopy|index_select|indexselect|index_put|indexput|"
                r"scatter|gather|cumsum|nonzero|where"),
    ("matmul", r"gemm|cutlass|cublas|xmma|sm\d\d_|ampere_|addmm|\bmm\b|matmul"),
    ("conv", r"conv"),
    ("norm_act", r"norm|silu|sigmoid|softplus|swish|gelu|rsqrt"),
    ("copy_elem", r"copy|fill|cat|stack|elementwise|vectorized|unrolled|reduce|mul|add"),
]


def categorise(name: str) -> str:
    n = name.lower()
    for cat, pat in CATEGORIES:
        if re.search(pat, n):
            return cat
    return "other"


def selftest() -> bool:
    cases = [
        ("chunk_delta_rule_fwd_kernel_h", "recurrence"),
        ("fwd_prepare_wy_repr_kernel_chunk64", "recurrence"),
        ("l2norm_fwd_kernel", "recurrence"),
        ("chunk_fwd_kernel_o", "recurrence"),
        ("void at::native::index_copy_kernel_impl<...>", "packing"),
        ("void at::native::(anonymous namespace)::indexSelectLargeIndex<...>", "packing"),
        ("ampere_bf16_s16816gemm_bf16_128x64_ldg8_f2f_stages_64x4_tn", "matmul"),
        ("sm80_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize", "matmul"),
        ("causal_conv1d_fwd_kernel", "conv"),
        ("void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor>", "copy_elem"),
        ("rms_norm_fwd_kernel", "norm_act"),
        ("zzz_unknown_kernel_xyz", "other"),
    ]
    ok = True
    print("profile_ragged self-test: kernel-name categoriser")
    for name, want in cases:
        got = categorise(name)
        good = got == want
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {name[:60]:<60} -> {got} (want {want})")
    # the test must be able to fail: a categoriser that maps everything to
    # 'other' would pass a test that only checked the unknown name
    lazy = sum(1 for n, w in cases if w != "other")
    ok &= lazy >= 10
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return ok


def profile(out_path: str):
    import torch
    from torch.profiler import ProfilerActivity, profile as tprofile

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "work"))
    from check_ragged import (K, VOCAB, build, orders, prefix_gates,  # noqa: E402
                              set_ragged, time_fn)
    from oracle_order import set_external_gates  # noqa: E402
    DEV = "cuda"

    def kernel_events(prof):
        evs = []
        for e in prof.key_averages():
            dt = getattr(e, "device_type", None)
            if dt is not None and str(dt).endswith("CUDA"):
                t = getattr(e, "self_device_time_total", None)
                if t is None:
                    t = getattr(e, "self_cuda_time_total", 0)
                evs.append((e.key, float(t), int(e.count)))
        return evs

    fixed4 = build(n_h=4, adaptive=False, n_heads=12, head_dim=32, hidden=128,
                   n_layers=1, vocab=VOCAB, seed=0).eval()
    adapt = build(n_h=K, adaptive=True, n_heads=12, head_dim=32, hidden=128,
                  n_layers=1, vocab=VOCAB, seed=0).eval()
    gen = torch.Generator(device=DEV).manual_seed(2)
    reps = 10
    results = []
    for B, T in [(32, 2048), (8, 2048)]:
        x = torch.randint(0, VOCAB, (B, T), generator=gen, device=DEV)
        g_mix = prefix_gates(orders("s5mix", B, T, gen))
        g_all = prefix_gates(orders("all4", B, T, gen))
        configs = [
            ("fixed4", fixed4, None, False),
            ("pathA_s5mix", adapt, g_mix, False),
            ("ragged_s5mix", adapt, g_mix, True),
            ("ragged_all4", adapt, g_all, True),
        ]
        print(f"\n=== batch {B}, length {T} ===")
        print(f"{'config':<14} {'wall':>7} {'kernels':>8} {'idle':>6} {'launch':>7} | "
              + " ".join(f"{c:>10}" for c, _ in CATEGORIES) + f" {'other':>7}")
        for name, mdl, gates, rag in configs:
            if gates is not None:
                set_external_gates(mdl, gates)
                set_ragged(mdl, rag)

            def fwd():
                return mdl(input_ids=x, use_cache=False)

            wall = time_fn(fwd)
            with torch.inference_mode():
                for _ in range(3):
                    fwd()
                torch.cuda.synchronize()
                with tprofile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    for _ in range(reps):
                        fwd()
                    torch.cuda.synchronize()
            evs = kernel_events(prof)
            per_cat = {c: 0.0 for c, _ in CATEGORIES}
            per_cat["other"] = 0.0
            others = {}
            for k, t, _ in evs:
                c = categorise(k)
                per_cat[c] += t / 1000.0 / reps          # ms per forward
                if c == "other":
                    others[k] = others.get(k, 0.0) + t / 1000.0 / reps
            total = sum(per_cat.values())
            launches = sum(cnt for _, _, cnt in evs) / reps
            top = sorted(((t / 1000.0 / reps, k) for k, t, _ in evs), reverse=True)[:12]
            print(f"{name:<14} {wall:7.2f} {total:8.2f} {max(0.0, wall - total):6.2f} "
                  f"{launches:7.0f} | " + " ".join(f"{per_cat[c]:10.2f}" for c, _ in CATEGORIES)
                  + f" {per_cat['other']:7.2f}")
            results.append(dict(batch=B, length=T, config=name, wall_ms=wall,
                                kernel_ms=total, launches_per_fwd=launches,
                                per_category_ms=per_cat,
                                other_kernels_ms=others,
                                top_kernels_ms=[[k, t] for t, k in top]))
            if gates is not None:
                set_external_gates(mdl, None)
                set_ragged(mdl, False)
        print("  top kernels, ragged_s5mix:")
        for k, t in [r for r in results if r["batch"] == B and r["config"] == "ragged_s5mix"][0]["top_kernels_ms"]:
            print(f"    {t:7.3f} ms  [{categorise(k):<10}] {k[:90]}")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(dict(device=torch.cuda.get_device_name(),
                                              reps=reps, rows=results), indent=2))
    print(f"\nwritten: {out_path}")
    print("All times in ms per forward pass. 'idle' = wall - kernel time (GPU waiting).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default="results/ragged_profile.json")
    args = ap.parse_args()
    ok = selftest()
    if args.selftest:
        sys.exit(0 if ok else 1)
    if not ok:
        print("categoriser self-test failed; refusing to profile")
        sys.exit(1)
    profile(args.out)


if __name__ == "__main__":
    main()
