"""
Heterogeneous-demand S5 word problems.

WHY
---
`--group=S5` samples uniformly from all 120 permutations. Transposition length
L runs 0..4, but the distribution is top-heavy -- 74 of 120 elements need L >= 3
-- so nearly every token is hard, demand is close to uniform, and there is very
little for an adaptive-order model to exploit. That is where the (H_n - 1)/(n-1)
= 32.1% ceiling comes from, and it is the ADVERSARIAL case for adaptive order.

This generator produces the other regime: mostly easy tokens (transpositions,
L=1), a fraction p of hard ones (5-cycles, L=4). Expected demand is

    E[L] = 1*(1-p) + 4*p = 1 + 3p

against a fixed-order model which must provision L_max = 4 for every token.

WHY THIS IS NOT A BENCHMARK BUILT TO WIN
----------------------------------------
The difficulty tiers are the authors' own. `generate_data.py` already ships
`S5_only_swaps` (transpositions only) and `limit_to` (permutations moving at
most k elements). We combine two of THEIR tiers with a mixing rate; we do not
invent a notion of difficulty. And p is swept across its whole range and
reported in full, including p=1 where adaptive order barely helps.

THE VOCABULARY TRAP (important)
-------------------------------
Mixing only L=1 and L=4 gives 34 distinct INPUT tokens, but the running products
span all 120. main.py builds its tokenizer from the input column alone, so 86
target values would silently become UNK.

The authors already solved this: main.py reads the vocabulary from `S5=2.csv`
when the group name contains "limit_to" or starts with "S5_only_swaps". So the
group names produced here contain "limit_to" and start with "S5", which routes
main.py to that branch (it takes g = group[:2]).

    PREREQUISITE:  S5=2.csv must exist.
        PYTHONPATH=$PWD python src/generate_data.py --group=S5 --k=2

TOKEN IDS
---------
main.py adds the group tokens first, sorted numerically, then the special
tokens. So token id == element index for 0..119, and specials occupy 120+.
That makes the per-token difficulty a direct lookup, which is what the oracle
experiment needs -- no extra CSV column, nothing for the collate function to
choke on. The lookup is written to a sidecar JSON.

USAGE
-----
    python work/gen_heterogeneous.py --p 0.1 --k 128 --samples 200000
    python work/gen_heterogeneous.py --sweep --k 128 --samples 200000
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import Counter
from pathlib import Path

N = 5                                   # S5
ELEMENTS = list(itertools.permutations(range(N)))
INDEX = {p: i for i, p in enumerate(ELEMENTS)}


# ---------------------------------------------------------------------------
# group machinery
# ---------------------------------------------------------------------------

def n_cycles(p: tuple[int, ...]) -> int:
    seen = [False] * len(p)
    c = 0
    for i in range(len(p)):
        if not seen[i]:
            c += 1
            j = i
            while not seen[j]:
                seen[j] = True
                j = p[j]
    return c


def transposition_length(p: tuple[int, ...]) -> int:
    """Cayley distance to the identity under transpositions: n - (number of cycles).

    Classical; see Feller, or Arratia-Barbour-Tavare. Equals the minimum number
    of generalized Householder factors needed to represent the transition
    (Grazzi et al., Thm 3; verified numerically in work/reference_deltaproduct.py).
    """
    return len(p) - n_cycles(p)


def compose(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    """Apply a, then b. Left-to-right, which matches reading a sequence."""
    return tuple(b[a[i]] for i in range(len(a)))


ELL = [transposition_length(p) for p in ELEMENTS]
IDENTITY = tuple(range(N))


def pool(ell: int) -> list[int]:
    return [i for i, e in enumerate(ELL) if e == ell]


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def make_sequences(p_hard, k, samples, easy_ell, hard_ell, seed):
    rng = random.Random(seed)
    easy, hard = pool(easy_ell), pool(hard_ell)
    if not easy or not hard:
        raise SystemExit(f"empty pool: |L={easy_ell}|={len(easy)}, |L={hard_ell}|={len(hard)}")

    rows, ell_hist = [], Counter()
    seen = set()
    attempts = 0
    while len(rows) < samples:
        attempts += 1
        if attempts > 20 * samples:
            raise SystemExit("could not find enough distinct sequences")
        seq = [rng.choice(hard) if rng.random() < p_hard else rng.choice(easy)
               for _ in range(k)]
        t = tuple(seq)
        if t in seen:
            continue
        seen.add(t)

        acc = IDENTITY
        out = []
        for tok in seq:
            acc = compose(acc, ELEMENTS[tok])
            out.append(INDEX[acc])
        ell_hist.update(ELL[tok] for tok in seq)
        rows.append((seq, out))
    return rows, ell_hist


def write(rows, ell_hist, group, k, data_dir, p_hard, easy_ell, hard_ell, seed):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{group}={k}.csv"

    with path.open("w") as f:
        f.write("seed,input,target\n")
        for seq, out in rows:
            f.write(f'{seed},"{" ".join(map(str, seq))}","{" ".join(map(str, out))}"\n')

    total = sum(ell_hist.values())
    emp = sum(e * c for e, c in ell_hist.items()) / total
    meta = {
        "group": group,
        "k": k,
        "n_sequences": len(rows),
        "p_hard": p_hard,
        "easy_ell": easy_ell,
        "hard_ell": hard_ell,
        "seed": seed,
        "ell_lookup": ELL,                   # element index -> transposition length
        "ell_histogram": {str(a): b for a, b in sorted(ell_hist.items())},
        "empirical_mean_ell": emp,
        "predicted_mean_ell": easy_ell + (hard_ell - easy_ell) * p_hard,
        "fixed_order_cost": hard_ell,        # a fixed model must provision L_max
        "adaptive_cost": emp,
        "cost_ratio": hard_ell / emp,
        "note": ("Fixed order is integer-valued, so for any budget B with "
                 f"{emp:.3f} <= B < {hard_ell} no fixed-order model both fits the "
                 "budget and tracks the group; adaptive order does."),
    }
    (data_dir / f"{group}_meta.json").write_text(json.dumps(meta, indent=2))
    return path, meta


# ---------------------------------------------------------------------------
# checks -- run before writing anything
# ---------------------------------------------------------------------------

def self_test() -> bool:
    ok = True
    print("=" * 70)
    print("SELF-TEST")
    print("=" * 70)

    hist = Counter(ELL)
    exp = {0: 1, 1: 10, 2: 35, 3: 50, 4: 24}
    t1 = dict(sorted(hist.items())) == exp
    print(f"  element counts by L         : {dict(sorted(hist.items()))}  "
          f"[{'PASS' if t1 else 'FAIL'}]")
    ok &= t1

    H = sum(1.0 / i for i in range(1, N + 1))
    mean = sum(ELL) / len(ELL)
    t2 = abs(mean - (N - H)) < 1e-12
    print(f"  E[L] == n - H_n             : {mean:.6f} vs {N - H:.6f}  "
          f"[{'PASS' if t2 else 'FAIL'}]")
    ok &= t2

    # group axioms on the composition we use
    rng = random.Random(0)
    t3 = True
    for _ in range(2000):
        a, b, c = (ELEMENTS[rng.randrange(120)] for _ in range(3))
        if compose(compose(a, b), c) != compose(a, compose(b, c)):
            t3 = False
        if compose(a, IDENTITY) != a or compose(IDENTITY, a) != a:
            t3 = False
    print(f"  associative + identity      : {t3}  [{'PASS' if t3 else 'FAIL'}]")
    ok &= t3

    # closure of the index map
    t4 = all(INDEX[compose(ELEMENTS[i], ELEMENTS[j])] < 120
             for i in range(0, 120, 7) for j in range(0, 120, 11))
    print(f"  products stay in 0..119     : {t4}  [{'PASS' if t4 else 'FAIL'}]")
    ok &= t4

    # transpositions alone must generate all of S5, or targets would not span
    # the full vocabulary and the task would be degenerate
    reach, frontier = {IDENTITY}, [IDENTITY]
    gens = [ELEMENTS[i] for i in pool(1)]
    while frontier:
        nxt = []
        for x in frontier:
            for g in gens:
                y = compose(x, g)
                if y not in reach:
                    reach.add(y); nxt.append(y)
        frontier = nxt
    t5 = len(reach) == 120
    print(f"  transpositions generate S5  : {len(reach)}/120  "
          f"[{'PASS' if t5 else 'FAIL'}]")
    ok &= t5

    # the mixture's expected demand must match 1 + 3p
    t6 = True
    for p in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0):
        rows, h = make_sequences(p, 64, 300, 1, 4, seed=1)
        emp = sum(a * b for a, b in h.items()) / sum(h.values())
        pred = 1 + 3 * p
        if abs(emp - pred) > 0.12:
            t6 = False
        print(f"    p={p:<5} predicted E[L]={pred:.3f}  empirical={emp:.3f}")
    print(f"  mixture matches 1 + 3p      : [{'PASS' if t6 else 'FAIL'}]")
    ok &= t6

    # targets must actually cover the whole group, else vocab is degenerate
    rows, _ = make_sequences(0.1, 128, 500, 1, 4, seed=2)
    tgt = {t for _, out in rows for t in out}
    t7 = len(tgt) == 120
    print(f"  targets cover all 120       : {len(tgt)}/120  "
          f"[{'PASS' if t7 else 'FAIL'}]")
    ok &= t7

    # inputs do NOT cover the vocabulary -- this is the trap the group name fixes
    inp = {t for seq, _ in rows for t in seq}
    print(f"  inputs cover only           : {len(inp)}/120  "
          f"<- why the group name must contain 'limit_to'")

    print("=" * 70)
    print("ALL SELF-TESTS PASSED" if ok else "SELF-TESTS FAILED")
    print("=" * 70)
    return ok


# ---------------------------------------------------------------------------

def group_name(p_hard: float) -> str:
    # must start with "S5" and contain "limit_to" so main.py takes the
    # vocabulary from S5=2.csv rather than from our restricted inputs
    return f"S5_limit_to_het_p{int(round(p_hard * 1000)):03d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, default=0.1, help="fraction of hard tokens")
    ap.add_argument("--k", type=int, default=128, help="sequence length")
    ap.add_argument("--samples", type=int, default=200000)
    ap.add_argument("--easy_ell", type=int, default=1)
    ap.add_argument("--hard_ell", type=int, default=4)
    ap.add_argument("--seed", type=int, default=666)
    ap.add_argument("--data_dir", default="state_tracking/data")
    ap.add_argument("--sweep", action="store_true",
                    help="generate p in {0, .05, .1, .25, .5, 1.0}")
    ap.add_argument("--skip_tests", action="store_true")
    args = ap.parse_args()

    if not args.skip_tests and not self_test():
        raise SystemExit("self-tests failed; nothing written")

    ps = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0] if args.sweep else [args.p]

    vocab_src = Path(args.data_dir) / "S5=2.csv"
    if not vocab_src.exists():
        print(f"\nWARNING: {vocab_src} not found. main.py needs it for the "
              f"vocabulary.\n  Generate it with:\n"
              f"    PYTHONPATH=$PWD python src/generate_data.py --group=S5 --k=2\n")

    print(f"\n{'group':<28} {'p':>5} {'E[L]':>7} {'fixed':>6} {'ratio':>7}  file")
    print("-" * 88)
    for p in ps:
        g = group_name(p)
        rows, hist = make_sequences(p, args.k, args.samples,
                                    args.easy_ell, args.hard_ell, args.seed)
        path, meta = write(rows, hist, g, args.k, args.data_dir,
                           p, args.easy_ell, args.hard_ell, args.seed)
        print(f"{g:<28} {p:>5} {meta['empirical_mean_ell']:>7.3f} "
              f"{meta['fixed_order_cost']:>6} {meta['cost_ratio']:>7.2f}x  {path.name}")

    print(f"\nTrain with:  --group={group_name(ps[0])} --k={args.k}")
    print("The oracle order for a token is meta['ell_lookup'][token_id] "
          "(token id == element index).")


if __name__ == "__main__":
    main()
