"""
Oracle order: force n_t = L(g_t) from ground truth, with no learning.

This is the upper bound for the matched-compute experiment. If the oracle
cannot track the group at cost E[L] while fixed order needs L_max, there is no
point training a gate to approximate it -- so this runs BEFORE the learned gate.

TOKEN ID == ELEMENT INDEX
-------------------------
main.py builds its tokenizer by adding the group tokens first, sorted
numerically, then the special tokens. So ids 0..|G|-1 are the group elements in
index order and the specials occupy the tail. That makes per-token difficulty a
direct lookup with no extra CSV column -- which matters, because an extra column
would reach `pad_collate`, which only knows how to stack input_ids and labels.

Special tokens (BOS, PAD, ...) get order 0: they carry no group element, so the
correct transition for them is the identity, which is exactly what a closed gate
produces (verified: beta=0 gives the identity factor, oracle test T1).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


def load_ell_lookup(meta_path: str | Path, n_vocab: int,
                    device=None, dtype=torch.long) -> torch.Tensor:
    """token id -> transposition length, as a length-n_vocab tensor.

    `meta_path` is the sidecar written by work/gen_heterogeneous.py.
    """
    meta = json.loads(Path(meta_path).read_text())
    ell = meta["ell_lookup"]
    if len(ell) > n_vocab:
        raise ValueError(f"lookup has {len(ell)} entries but n_vocab is {n_vocab}")
    t = torch.zeros(n_vocab, dtype=dtype)
    t[: len(ell)] = torch.tensor(ell, dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t


def oracle_gates(input_ids: torch.Tensor, ell_lookup: torch.Tensor,
                 max_order: int, min_order: int = 0,
                 dtype: torch.dtype | None = None) -> torch.Tensor:
    """(B, T) token ids -> (B, T, K) hard gates, monotone by construction.

    gate[b, t, i] = 1 iff i < clamp(L(token), min_order, max_order)

    so the effective order of token t is exactly its transposition length,
    clipped to what the model can express.
    """
    dtype = dtype or torch.get_default_dtype()
    ell = ell_lookup.to(input_ids.device)[input_ids]
    ell = ell.clamp(min=min_order, max=max_order)
    idx = torch.arange(max_order, device=input_ids.device)
    return (idx.view(1, 1, -1) < ell.unsqueeze(-1)).to(dtype)


def set_external_gates(model, gates: torch.Tensor | None) -> int:
    """Install gates on every adaptive-order layer. Returns how many were set.

    Pass None to clear. The layer uses these in place of its halting gate, which
    is what makes the oracle arm and the learned arm otherwise identical.
    """
    n = 0
    for m in model.modules():
        if hasattr(m, "adaptive_order") and hasattr(m, "external_gates"):
            m.external_gates = gates
            n += 1
    if n == 0:
        raise RuntimeError(
            "no adaptive-order layers found. Was the model built with "
            "adaptive_order=True, and is the layer patch applied?"
        )
    return n


def effective_cost(input_ids: torch.Tensor, ell_lookup: torch.Tensor,
                   max_order: int, ignore_ids: torch.Tensor | None = None) -> float:
    """Mean Householder factors per token under the oracle -- the x-axis of the
    matched-compute plot. Excludes special tokens, which are free."""
    ell = ell_lookup.to(input_ids.device)[input_ids].clamp(max=max_order).float()
    if ignore_ids is not None:
        keep = ~torch.isin(input_ids, ignore_ids.to(input_ids.device))
        return ell[keep].mean().item() if keep.any() else 0.0
    return ell.mean().item()


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def _tests() -> bool:
    import itertools, tempfile

    N = 5
    els = list(itertools.permutations(range(N)))

    def cyc(p):
        seen = [False] * len(p); c = 0
        for i in range(len(p)):
            if not seen[i]:
                c += 1; j = i
                while not seen[j]:
                    seen[j] = True; j = p[j]
        return c

    ELL = [N - cyc(p) for p in els]
    n_vocab = 127                      # 120 group tokens + 7 specials

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"ell_lookup": ELL}, f)
        meta = f.name

    ok = True
    print("=" * 70)
    print("Oracle order — tests")
    print("=" * 70)

    lut = load_ell_lookup(meta, n_vocab)
    t1 = (lut.shape[0] == n_vocab and lut[:120].tolist() == ELL
          and lut[120:].sum().item() == 0)
    print(f"T1  lookup shape + specials are order 0 : {t1}  "
          f"[{'PASS' if t1 else 'FAIL'}]")
    ok &= t1

    # a transposition must get order 1, a 5-cycle order 4
    i_t = ELL.index(1)
    i_5 = ELL.index(4)
    ids = torch.tensor([[i_t, i_5, 120, i_t]])          # 120 = a special token
    g = oracle_gates(ids, lut, max_order=4)
    n_t = g.sum(-1)
    t2 = n_t.tolist() == [[1.0, 4.0, 0.0, 1.0]]
    print(f"T2  order == transposition length       : {n_t.tolist()[0]}  "
          f"[{'PASS' if t2 else 'FAIL'}]")
    ok &= t2

    mono = bool((g[..., :-1] - g[..., 1:] >= 0).all())
    binary = bool(((g == 0) | (g == 1)).all())
    t3 = mono and binary
    print(f"T3  gates monotone and binary           : {mono}, {binary}  "
          f"[{'PASS' if t3 else 'FAIL'}]")
    ok &= t3

    # clipping: with max_order=2 a 5-cycle must saturate, not overflow
    g2 = oracle_gates(ids, lut, max_order=2)
    t4 = g2.sum(-1).tolist() == [[1.0, 2.0, 0.0, 1.0]]
    print(f"T4  clips to max_order                  : {g2.sum(-1).tolist()[0]}  "
          f"[{'PASS' if t4 else 'FAIL'}]")
    ok &= t4

    # cost must reproduce 1 + 3p on a mixture
    torch.manual_seed(0)
    t5 = True
    print("T5  mean cost tracks 1 + 3p")
    for p in (0.0, 0.1, 0.25, 0.5, 1.0):
        easy = [i for i, e in enumerate(ELL) if e == 1]
        hard = [i for i, e in enumerate(ELL) if e == 4]
        toks = []
        for _ in range(8000):
            pool = hard if torch.rand(1).item() < p else easy
            toks.append(pool[torch.randint(len(pool), (1,)).item()])
        c = effective_cost(torch.tensor(toks).view(1, -1), lut, 4)
        pred = 1 + 3 * p
        good = abs(c - pred) < 0.12
        t5 &= good
        print(f"      p={p:<5} predicted {pred:.3f}  measured {c:.3f}  "
              f"{'ok' if good else 'MISMATCH'}")
    print(f"    [{'PASS' if t5 else 'FAIL'}]")
    ok &= t5

    print("=" * 70)
    print("ALL TESTS PASSED" if ok else "TESTS FAILED")
    print("=" * 70)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _tests() else 1)
