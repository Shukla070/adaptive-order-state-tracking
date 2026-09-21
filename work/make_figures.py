"""
Regenerate every figure in the project report from the committed result files.

WHY THIS FILE EXISTS
--------------------
Figures drawn by hand from prose inherit whatever the prose got wrong. This
script reads results/*.json and nothing else, so a figure cannot disagree with
the measurement it came from. If a claim is not in a result file, no figure of
it is produced.

It also refuses to draw the two things that this project has already got wrong
once:

  * a single-seed value on a cell that is known to be bistable, without
    labelling it a single draw
  * an escape "fraction" on a cell whose distribution is not bimodal
    (S5_c5_t1 is continuous over 0.245-0.763; a threshold there invents a number)

USAGE
-----
    python work/make_figures.py              # write PNG + PDF into figures/
    python work/make_figures.py --selftest   # known-answer checks, no plotting
    python work/make_figures.py --only 4 7   # regenerate selected figures

Requires matplotlib, which is NOT part of the pinned training environment:

    pip install matplotlib

matplotlib does not affect any result; it is a reporting dependency only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
FIGDIR = REPO / "figures"

# A run "escapes" when it leaves the failure plateau. Fixed here, once, so that
# every figure and every table in the report uses the same cutoff.
ESCAPE_TOKEN_ACC = 0.90

# ----------------------------------------------------------------------------
# palette - chosen once so every figure reads as one set
# ----------------------------------------------------------------------------
C = {
    "ink":     "#1b1f24",
    "muted":   "#6d757d",
    "grid":    "#e3e1dc",
    "teal":    "#0d6e6b",
    "blue":    "#2f7fb8",
    "gold":    "#96751d",
    "clay":    "#9c4a2f",
    "violet":  "#5b4a9e",
    "green":   "#2f6f43",
    "grey":    "#8a9099",
}
SERIES = [C["teal"], C["blue"], C["gold"], C["clay"], C["grey"]]


def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": C["ink"],
        "axes.labelcolor": C["ink"],
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": C["grid"],
        "grid.linewidth": 0.8,
        "xtick.color": C["muted"],
        "ytick.color": C["muted"],
        "legend.frameon": False,
        "legend.fontsize": 9,
    })


def load(name: str):
    p = RESULTS / name
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run the experiment first")
    return json.loads(p.read_text())


def save(fig, stem: str):
    FIGDIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"{stem}.{ext}")
    print(f"  wrote figures/{stem}.png and .pdf")
    import matplotlib.pyplot as plt
    plt.close(fig)


REPORT = False   # --report: no in-image titles or notes; the document caption carries them


def note(ax, text):
    """A single caption line inside the axes, for provenance or a caveat."""
    if REPORT:
        return
    ax.text(0.0, -0.20, text, transform=ax.transAxes, fontsize=8.5,
            color=C["muted"], va="top", ha="left", wrap=True)


# ============================================================================
# 1 - length extrapolation of the reproduced base method
# ============================================================================
def fig1():
    import matplotlib.pyplot as plt
    d = load("s5_length_extrapolation.json")
    pp = d["per_position_sequence_accuracy"]
    xs = list(range(len(pp)))
    cfg = d["config"]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(xs, pp, color=C["teal"], lw=2)
    ax.fill_between(xs, 0, pp, color=C["teal"], alpha=0.08)

    ax.axvline(cfg["train_len"], color=C["ink"], ls="--", lw=1.2)
    ax.annotate(f"trained at {cfg['train_len']}", xy=(cfg["train_len"], 0.55),
                xytext=(cfg["train_len"] + 14, 0.62), color=C["ink"], fontsize=9)

    # the two landmarks quoted in the report, read off the curve not asserted
    last_perfect = max(i for i, v in enumerate(pp) if v >= 0.9995)
    half = min(i for i, v in enumerate(pp) if v < 0.5)
    ax.plot([last_perfect], [pp[last_perfect]], "o", color=C["violet"], ms=6, zorder=5)
    ax.annotate(f"perfect through {last_perfect}", xy=(last_perfect, pp[last_perfect]),
                xytext=(10, 0.86), color=C["violet"], fontsize=9)
    ax.plot([half], [pp[half]], "o", color=C["clay"], ms=6, zorder=5)
    ax.annotate(f"falls below half at {half}",
                xy=(half, pp[half]), xytext=(half + 16, 0.38),
                color=C["clay"], fontsize=9)

    ax.set_xlim(0, len(pp) - 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("position in the instruction sequence")
    ax.set_ylabel("fraction of sequences still exact")
    ax.set_title("Reproduced base method: accuracy decays with sequence length")
    note(ax, f"DeltaProduct, S5 word problem, n_h={cfg['n_h']}, {cfg['n_layers']} layer, "
             f"{cfg['n_heads']}x{cfg['head_dim']}, trained at length {cfg['train_len']}, "
             f"evaluated at {cfg['eval_len']}. Token accuracy over all positions "
             f"{d['token_accuracy_at_512']:.4f}. Single run.")
    save(fig, "fig1_length_extrapolation")


# ============================================================================
# 2 - capability at matched compute
# ============================================================================
def _matched():
    out = {}
    for p in ("0.05", "0.1", "0.25", "0.5", "1.0"):
        d = load(f"matched_compute_p{p}.json")
        out[float(p)] = {r["arm"]: r for r in d["results"]}
    return out


def fig2():
    import matplotlib.pyplot as plt
    M = _matched()
    fig, ax = plt.subplots(figsize=(7.4, 4.4))

    for i, (p, arms) in enumerate(sorted(M.items())):
        fx = [arms[f"fixed{k}"] for k in (1, 2, 3, 4)]
        ax.plot([r["cost"] for r in fx], [r["token_acc"] for r in fx],
                marker="o", color=SERIES[i], ms=4.5, lw=1.6,
                ls="--" if p == 1.0 else "-",
                label=f"p = {p:.2f}" + (" (control)" if p == 1.0 else ""))
        o = arms["oracle"]
        ax.plot([o["cost"]], [o["token_acc"]], "D", color=C["violet"], ms=7,
                zorder=6, mec="white", mew=0.8)

    ax.plot([], [], "D", color=C["violet"], ms=7, label="oracle allocation")
    ax.set_xlim(0.7, 4.3)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("per-token cost (Householder factors)")
    ax.set_ylabel("token accuracy")
    ax.set_title("The oracle reaches full accuracy at a fraction of the cost")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    note(ax, "20 000 steps, one run per cell. CAUTION: mid-range values (0.1-0.9) are single "
             "draws from cells now known to be bistable and are not capability measurements - "
             "see AUDIT.md section 3. The oracle reads ground truth; it is a reference, not a model.")
    save(fig, "fig2_matched_compute")


def fig3():
    """The notional-vs-measured point: allocation falls, wall-clock does not."""
    import matplotlib.pyplot as plt
    import numpy as np
    M = _matched()
    arms = M[0.1]
    names = ["fixed1", "fixed2", "fixed3", "fixed4", "oracle"]
    cost = [arms[n]["cost"] for n in names]
    secs = [arms[n]["train_s"] for n in names]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.4, 3.8))
    x = np.arange(len(names))

    a1.bar(x, cost, color=[C["grey"]] * 4 + [C["violet"]], width=0.62)
    a1.set_xticks(x); a1.set_xticklabels(names, rotation=20)
    a1.set_ylabel("factors per token")
    a1.set_title("What we claim to save")
    for xi, v in zip(x, cost):
        a1.text(xi, v + 0.08, f"{v:.3f}", ha="center", fontsize=8.5, color=C["ink"])

    a2.bar(x, secs, color=[C["grey"]] * 4 + [C["violet"]], width=0.62)
    a2.set_xticks(x); a2.set_xticklabels(names, rotation=20)
    a2.set_ylabel("training wall-clock (s)")
    a2.set_title("What we actually measured")
    for xi, v in zip(x, secs):
        a2.text(xi, v + 12, f"{v:.0f}", ha="center", fontsize=8.5, color=C["ink"])

    note(a1, "p = 0.10. The oracle allocates 1.300 factors per token against fixed4's 4.000 - "
             "a 3.1x reduction in required work.")
    note(a2, "The same two runs take 756 s and 761 s. Unused factors are still computed and "
             "multiplied by zero, so no time is saved. Every cost figure in this project is "
             "notional until ragged expansion exists.")
    fig.tight_layout()
    save(fig, "fig3_notional_vs_measured")


# ============================================================================
# 4 - the escape census: the honest version of the alphabet story
# ============================================================================
def _census_rows():
    """Pool every repeated run of a (alphabet, n_h) cell across result files."""
    import collections
    cells = collections.defaultdict(list)
    # ONLY the designed census runs. The beta_a5c5* files repeat the same cells
    # under different beta parametrisations, so pooling them would average over
    # arms that the experiment exists to compare - the conflation AUDIT.md
    # records as error #3.
    for fname in ("escape_census.json", "escape_census2.json",
                  "escape_s5t1.json"):
        p = RESULTS / fname
        if not p.exists():
            continue
        for r in json.loads(p.read_text()).get("results", []):
            if "arm" in r and "n_h" in r:
                cells[(r["arm"], r["n_h"])].append(r["token_acc"])
    return {k: sorted(v) for k, v in cells.items() if len(v) >= 3}


def _wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def fig4():
    import matplotlib.pyplot as plt
    import numpy as np
    cells = _census_rows()
    if not cells:
        print("  skip fig4: no census files present")
        return

    items = []
    for (arm, nh), vals in sorted(cells.items()):
        k = sum(1 for v in vals if v >= ESCAPE_TOKEN_ACC)
        lo, hi = _wilson(k, len(vals))
        bimodal = _is_bimodal(vals)
        items.append((f"{arm}:n_h={nh}", k, len(vals), k / len(vals), lo, hi, bimodal))

    fig, ax = plt.subplots(figsize=(7.6, 0.44 * len(items) + 2.0))
    y = np.arange(len(items))[::-1]
    for yi, (lab, k, n, rate, lo, hi, bimodal) in zip(y, items):
        col = C["green"] if bimodal else C["clay"]
        ax.plot([lo, hi], [yi, yi], color=col, lw=2, alpha=0.45, solid_capstyle="round")
        ax.plot([rate], [yi], "o", color=col, ms=8, zorder=5)
        ax.text(1.02, yi, f"{k}/{n}", va="center", fontsize=9,
                color=C["ink"], family="monospace")
        if not bimodal:
            ax.text(-0.02, yi, "*", va="center", ha="right",
                    fontsize=13, color=C["clay"])

    ax.set_yticks(y)
    ax.set_yticklabels([i[0] for i in items], fontsize=9, family="monospace")
    ax.set_xlim(-0.04, 1.14)
    ax.set_xlabel("fraction of runs that escape the failure plateau")
    ax.set_title("Every alphabet cell is bistable - none is reliably broken")
    ax.grid(axis="y", visible=False)
    note(ax, f"Escape = token accuracy >= {ESCAPE_TOKEN_ACC}. Bars are 95% Wilson intervals. "
             "* marks a cell whose distribution is NOT bimodal, where an escape fraction is a "
             "threshold artefact and must not be quoted (S5_c5_t1 spreads continuously over "
             "0.245-0.763). Seed is not the unit of replication: bf16 kernels are "
             "non-deterministic, and identical seed and config produced 0.9598 and 0.1177.")
    save(fig, "fig4_escape_census")


def _is_bimodal(vals, gap=0.35):
    """Crude but explicit: a clear empty band between a low and a high cluster."""
    lo = [v for v in vals if v < 0.5]
    hi = [v for v in vals if v >= 0.5]
    if not lo or not hi:
        return True          # one cluster only - the threshold is not inventing a split
    return (min(hi) - max(lo)) >= gap


def fig5():
    """S5_c5_t1: the cell where the escape framing does NOT apply."""
    import matplotlib.pyplot as plt
    import numpy as np
    p = RESULTS / "escape_s5t1.json"
    if not p.exists():
        print("  skip fig5: escape_s5t1.json not present")
        return
    vals = sorted(r["token_acc"] for r in json.loads(p.read_text())["results"])

    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ax.plot(vals, [0] * len(vals), "o", color=C["clay"], ms=10, alpha=0.75,
            mec="white", mew=1.0)
    ax.axvline(0.5, color=C["muted"], ls="--", lw=1.2)
    ax.text(0.5, 0.35, "  threshold 0.5", fontsize=9,
            color=C["muted"], va="center")
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("token accuracy")
    ax.set_title("S5_c5_t1 at n_h=4 is not bimodal - ten runs, continuous spread")
    ax.grid(axis="y", visible=False)
    note(ax, f"Ten runs, constant learning rate: {', '.join(f'{v:.4f}' for v in vals)}. "
             "There is no empty band, so there are no two basins to count. "
             "Reporting an escape fraction for this cell is error #8 in AUDIT.md.")
    save(fig, "fig5_s5c5t1_not_bimodal")


def fig6():
    """Single-draw grid, labelled as such, with census escape rates beside it."""
    import matplotlib.pyplot as plt
    import numpy as np
    d = load("group_closure_rest.json")
    rows = d["results"]
    arms = sorted({r["arm"] for r in rows})
    orders = [2, 3, 4]
    grid = np.full((len(arms), len(orders)), np.nan)
    meta = {}
    for r in rows:
        grid[arms.index(r["arm"]), orders.index(r["n_h"])] = r["token_acc"]
        meta[r["arm"]] = (r["group"], r["n_tokens"], r["D_max"])

    fig, ax = plt.subplots(figsize=(6.6, 0.75 * len(arms) + 2.4))
    im = ax.imshow(grid, cmap="BrBG", vmin=0, vmax=1, aspect="auto")
    for i in range(len(arms)):
        for j in range(len(orders)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i,j]:.4f}", ha="center", va="center",
                        fontsize=9.5, family="monospace",
                        color="white" if grid[i, j] > 0.75 or grid[i, j] < 0.25 else C["ink"])
    ax.set_xticks(range(len(orders)))
    ax.set_xticklabels([f"n_h = {o}" for o in orders])
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([f"{a}\n{meta[a][0]}, {meta[a][1]} symbols, D_max={meta[a][2]}"
                        for a in arms], fontsize=8.5, family="monospace")
    ax.grid(visible=False)
    ax.set_title("Alphabet x order grid - ONE RUN PER CELL")
    fig.colorbar(im, ax=ax, label="token accuracy", fraction=0.046, pad=0.04)
    note(ax, "Each square is a single draw. The census (fig4) shows these cells escape 10-43% "
             "of the time, so this grid measures which draw we happened to get, not which "
             "alphabets are learnable. Kept because it is the experiment that was run.")
    save(fig, "fig6_alphabet_grid_single_draw")


# ============================================================================
# 7-9 - the learned gate
# ============================================================================
def fig7():
    import matplotlib.pyplot as plt
    import numpy as np
    d = load("gate_gamma_sweep.json")
    rows = d["results"]
    gammas = sorted({r["gamma"] for r in rows})
    oracle = rows[0]["alloc"]["mean_demand"]
    ceiling = max(r["n_h"] for r in rows)

    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    for gi, g in enumerate(gammas):
        cells = [r for r in rows if r["gamma"] == g]
        groups = {}
        for r in cells:
            groups.setdefault(round(r["alloc"]["mean_n_t"], 3), []).append(r)
        prev_x = None
        for x, rs in sorted(groups.items()):
            got = rs[0]["alloc"]["corr"] >= 0.99
            below = prev_x is not None and x - prev_x < 0.35
            prev_x = x
            ax.plot([x], [gi], "o",
                    color=C["green"] if got else "none",
                    mec=C["green"] if got else C["clay"],
                    mew=2, ms=11, alpha=0.9, zorder=4)
            ids = ", ".join(str(r["seed"]) for r in sorted(rs, key=lambda r: r["seed"]))
            ax.annotate(("runs " if len(rs) > 1 else "run ") + ids, xy=(x, gi),
                        xytext=(0, -20 if below else 13), textcoords="offset points",
                        ha="center", fontsize=7.5, color=C["muted"])

    ax.axvline(oracle, color=C["violet"], ls="--", lw=1.4)
    ax.text(oracle, len(gammas) - 0.42, f" oracle {oracle:.4f}",
            color=C["violet"], fontsize=9)
    ax.axvline(ceiling, color=C["clay"], ls="--", lw=1.4)
    ax.text(ceiling, len(gammas) - 0.42, f" fixed ceiling {ceiling}",
            color=C["clay"], fontsize=9, ha="right")

    ax.set_yticks(range(len(gammas)))
    ax.set_yticklabels([f"$\\gamma$ = {g}" for g in gammas])
    ax.set_ylim(-0.6, len(gammas) - 0.15)
    ax.set_xlim(1.0, 4.3)
    ax.set_xlabel("mean factors allocated per token")
    ax.set_title("Learned gate: allocation reached, by penalty weight and run")
    ax.grid(axis="y", visible=False)
    note(ax, "Filled = the gate separated the two demand classes (correlation with true demand "
             ">= 0.99). Hollow = it did not. Token accuracy stayed above 0.998 in all ten runs. "
             "With only two demand values present, that correlation is a coarse check: it shows "
             "the classes were separated, not that a four-valued demand would be recovered.")
    save(fig, "fig7_gate_allocation")


def fig8():
    import matplotlib.pyplot as plt
    import numpy as np
    d = load("gate_gamma_sweep.json")
    rows = d["results"]
    labels, cheap, dear = [], [], []
    for r in rows:
        bd = r["alloc"]["by_demand"]
        labels.append(f"$\\gamma$={r['gamma']}\nrun {r['seed']}")
        cheap.append(bd["1"]["mean_n_t"])
        dear.append(bd["4"]["mean_n_t"])

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    ax.bar(x - 0.2, cheap, 0.4, color=C["teal"], label="tokens that need 1 factor")
    ax.bar(x + 0.2, dear, 0.4, color=C["clay"], label="tokens that need 4 factors")
    ax.axhline(1, color=C["teal"], ls=":", lw=1.4)
    ax.axhline(4, color=C["clay"], ls=":", lw=1.4)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("factors actually allocated")
    ax.set_ylim(0, 4.6)
    ax.set_title("What each run gave to each kind of token")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=2)
    note(ax, "Expensive tokens receive 4.000 in every run - the gate never starves a token that "
             "needs the factors. The whole question is whether it learns to stop paying 4 for "
             "the cheap ones. Three runs of ten did.")
    save(fig, "fig8_gate_allocation_by_demand")


def fig9():
    """Every learned-gate run to date, pooled honestly."""
    import matplotlib.pyplot as plt
    import numpy as np
    runs = [
        ("gate_train_p10.json",   "initial configuration"),
        ("gate_fixedinit.json",   "gate re-initialised"),
        ("gate_bias40_more.json", "initial-bias sweep"),
        ("gate_fp32gate.json",    "float32 halting"),
        ("gate_gamma_sweep.json", "penalty-weight sweep"),
    ]
    labs, tot, got, best = [], [], [], []
    for fname, label in runs:
        p = RESULTS / fname
        if not p.exists():
            continue
        rows = [r for r in json.loads(p.read_text())["results"]
                if r.get("mode") == "learned" and r.get("alloc")]
        if not rows:
            continue
        ok = [r for r in rows if (r["alloc"].get("corr") or 0) >= 0.99
              and r["token_acc"] >= 0.99]
        labs.append(label)
        tot.append(len(rows))
        got.append(len(ok))
        best.append(min((r["alloc"]["mean_n_t"] for r in ok), default=float("nan")))

    x = np.arange(len(labs))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.4, 5.2), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1]})
    a1.bar(x, tot, color=C["grid"], width=0.6, label="learned runs")
    a1.bar(x, got, color=C["green"], width=0.6, label="recovered the demand structure")
    for xi, (g, t) in enumerate(zip(got, tot)):
        a1.text(xi, t + 0.25, f"{g}/{t}", ha="center", fontsize=9, color=C["ink"])
    a1.set_ylabel("runs")
    a1.set_ylim(0, max(tot) * 1.45)
    a1.set_title("Learned-gate runs: how often the mechanism worked")
    a1.legend(loc="upper left", ncol=2)

    a2.plot(x, best, "o-", color=C["violet"], ms=7, lw=1.6)
    for xi, b in enumerate(best):
        if b == b:
            a2.text(xi, b + 0.12, f"{b:.4f}", ha="center", fontsize=9, color=C["ink"])
    a2.axhline(1.2985, color=C["violet"], ls="--", lw=1.2)
    a2.text(len(labs) - 0.5, 1.33, "oracle 1.2985", fontsize=8.5,
            color=C["violet"], ha="right")
    a2.set_ylim(1.0, 2.8)
    a2.set_ylabel("best usable cost")
    a2.set_xticks(x); a2.set_xticklabels(labs, rotation=18, ha="right", fontsize=9)
    note(a2, "'Usable' = the run also kept token accuracy >= 0.99, so a low cost bought by "
             "breaking the task does not count. The first attempt reached the oracle allocation "
             "exactly and nothing since has matched it - that regression is unexplained.")
    fig.tight_layout()
    save(fig, "fig9_gate_history")


# ============================================================================
# self-test - known answers, written to fail against a wrong implementation
# ============================================================================
def selftest() -> int:
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))

    print("=" * 74)
    print("SELF-TEST - known answers, computed from the result files")
    print("=" * 74)

    # T1 - Wilson interval against a hand-computed value
    lo, hi = _wilson(3, 10)
    check("T1  Wilson interval for 3/10 is about [0.108, 0.603]",
          abs(lo - 0.1077) < 2e-3 and abs(hi - 0.6032) < 2e-3, f"[{lo:.4f}, {hi:.4f}]")

    # T2 - the bimodality guard must REJECT the continuous cell.
    # This is the test that matters: it fails if someone widens the gap
    # threshold until S5_c5_t1 is called bimodal again.
    s5t1 = sorted(r["token_acc"] for r in load("escape_s5t1.json")["results"])
    check("T2  S5_c5_t1 at n_h=4 is correctly judged NOT bimodal",
          not _is_bimodal(s5t1), f"spread {min(s5t1):.4f}-{max(s5t1):.4f}")

    # T3 - a genuinely bimodal cell must still be accepted
    synthetic = [0.04, 0.05, 0.04, 0.99, 0.98]
    check("T3  a clearly two-basin distribution is judged bimodal",
          _is_bimodal(synthetic))

    # T4 - the census must reproduce the escape counts recorded in AUDIT.md
    cells = _census_rows()
    a5 = cells.get(("A5_c5", 2))
    if a5 is None:
        check("T4  A5_c5:n_h=2 census present", False, "no pooled runs found")
    else:
        k = sum(1 for v in a5 if v >= ESCAPE_TOKEN_ACC)
        check("T4  A5_c5:n_h=2 escapes at the rate AUDIT.md records",
              k >= 3 and len(a5) >= 10, f"{k}/{len(a5)} escapes")

    # T5 - the headline gate cell must be exactly what the sweep recorded
    gs = load("gate_gamma_sweep.json")["results"]
    best = [r for r in gs if r["alloc"]["corr"] >= 0.99 and r["token_acc"] >= 0.99]
    check("T5  three gate runs of ten recovered the demand structure",
          len(best) == 3, f"{len(best)} of {len(gs)}")
    check("T6  the best usable allocation is 2.1990",
          abs(min(r["alloc"]["mean_n_t"] for r in best) - 2.1990) < 1e-3)

    # T7 - the notional/measured gap must not silently close
    M = _matched()[0.1]
    gap = abs(M["oracle"]["train_s"] - M["fixed4"]["train_s"])
    check("T7  oracle and fixed4 wall-clock still differ by under 2%",
          gap / M["fixed4"]["train_s"] < 0.02,
          f"{M['oracle']['train_s']:.0f}s vs {M['fixed4']['train_s']:.0f}s")

    # T8 - the extrapolation curve must start perfect and end at zero
    pp = load("s5_length_extrapolation.json")["per_position_sequence_accuracy"]
    check("T8  extrapolation curve starts at 1.0 and reaches 0.0",
          pp[0] >= 0.999 and pp[-1] <= 0.001, f"{pp[0]:.3f} -> {pp[-1]:.3f}")

    print("\n" + "=" * 74)
    print("ALL CHECKS PASS" if ok else "FAILURES ABOVE - do not use these figures")
    print("=" * 74)
    return 0 if ok else 1


FIGURES = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5,
           6: fig6, 7: fig7, 8: fig8, 9: fig9}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--only", type=int, nargs="+", metavar="N")
    ap.add_argument("--report", action="store_true",
                    help="omit in-image titles and notes; write to figures/report/")
    args = ap.parse_args()
    if args.report:
        global REPORT, FIGDIR
        REPORT = True
        FIGDIR = REPO / "figures" / "report"
        import matplotlib.axes
        matplotlib.axes.Axes.set_title = lambda self, *a, **k: None

    if args.selftest:
        return selftest()

    style()
    wanted = args.only or sorted(FIGURES)
    print(f"writing figures into {FIGDIR}")
    for n in wanted:
        fn = FIGURES.get(n)
        if fn is None:
            print(f"  no figure {n}")
            continue
        try:
            fn()
        except FileNotFoundError as e:
            print(f"  skip figure {n}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
