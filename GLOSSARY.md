# Glossary — every term, symbol and setting used in this project

**Adaptive-Order State Transitions in Linear RNNs for Efficient State Tracking**
Utkarsh Shukla · NITK Surathkal · Guide: Anand Kumar M

Written so that someone who has never read the paper can follow any discussion,
log file or result in this repository. Read Part 1 and you can read the result
tables. Read Parts 2–4 and you can read the code.

---

## Part 0 — The running example

Keep this picture in mind for the whole document.

> Five cups sit in a row. You are handed instructions one at a time:
> *"swap cups 2 and 4"*, *"move 1→3, 3→5, 5→2, 2→4, 4→1"*, and so on.
> After **every** instruction you must say the current arrangement of all five
> cups.

That is the entire task. Everything below is vocabulary for describing the
instructions, the machine that follows them, and how much work each instruction
costs it.

---

## Part 1 — The task and the data

### State tracking
Reading symbols one at a time, where each symbol changes a hidden state, and
after every symbol you must report the current state. The difficulty is that the
answer at step 100 depends on all 100 symbols — you cannot look at symbol 100
alone.

### Why we use cup-shuffling specifically
Because it is the cleanest known test of whether a model is *genuinely* keeping
state rather than pattern-matching. There is no shortcut: get one instruction
wrong and every answer afterwards is wrong. It is also exactly solvable on
paper, so we always know the correct answer and can say precisely how much work
is required.

### Group
A set of operations where: doing two in a row gives you another operation from
the same set; there is a "do nothing" operation; and every operation can be
undone. Cup rearrangements form a group.

### Element
One specific operation. *"Swap cups 2 and 4"* is one element.

### S₅ — the symmetric group on 5 items
**All** 120 possible rearrangements of 5 cups. `S₃` = all 6 rearrangements of 3
items, `S₄` = 24, and so on. The subscript is the number of items.

### A₅ — the alternating group on 5 items
The **"even" half** of S₅ — 60 of the 120. Which half, and why it matters, is
explained under *parity* below. `A₄` = 12, `A₃` = 3.

> **This distinction turned out to be the central discovery of the project.**
> A₅ is not merely "smaller than S₅" — it is *structurally* cheaper for the
> model to handle. See Part 4.

### Transposition (a "swap")
An element that exchanges exactly two items and leaves the rest alone. There
are 10 of them in S₅. These are the *simplest* possible instructions.

### Cycle, and 5-cycle
An element that moves items around a loop. A **5-cycle** sends all five cups
around one loop (1→3→5→2→4→1). There are 24 of them in S₅. These are the
*hardest* instructions in our tasks.

### ℓ (ell) — "transposition length"
**The fewest simple swaps needed to build an element.** A swap has ℓ = 1. A
5-cycle has ℓ = 4 (you need four swaps to produce it). Doing nothing has ℓ = 0.

Formally ℓ(g) = n − (number of loops in g). For years this has been treated as
"how much work this instruction costs". **Part 4 shows that is not correct.**

### Parity — even and odd
Whether ℓ is an even or odd number. A swap (ℓ=1) is **odd**; a 5-cycle (ℓ=4) is
**even**. The even elements form Aₙ. This matters because combining two even
elements always gives an even one — so **even elements alone can never produce
an odd one**.

### Alphabet
The set of instructions a particular task actually uses. Not the whole group —
just the symbols that appear in the data.

### Closure, written ⟨A⟩
Everything you can build by chaining alphabet symbols together. The key fact
this project turns on:

| alphabet | closure | size |
|---|---|---|
| the 24 five-cycles | **A₅** (even only) | 60 |
| those same 24 **plus one single swap** | **S₅** (everything) | 120 |

One extra instruction — the simplest one that exists — doubles the world the
model must be able to represent.

### p — the "hard token rate"
In the data we generate, the fraction of instructions that are hard (5-cycles)
rather than easy (swaps). `p = 0.1` means 10% of instructions are 5-cycles and
90% are swaps. `p = 1.0` means every instruction is a 5-cycle.

### E[ℓ] — average work per instruction
The mean of ℓ across the data: **E[ℓ] = 1 + 3p** in our mixture (an easy
instruction costs 1, a hard one costs 4). At p=0.1 that is 1.3, versus the 4 a
fixed-budget model must reserve.

---

## Part 2 — The model

### Linear RNN
A model that keeps a running state and updates it with a fixed rule at each
step, rather than re-reading the whole input like a transformer. Cheap — cost
grows linearly with length, not quadratically — but historically much weaker at
state tracking. Recent work closed much of that gap; this project is about the
cost of doing so.

### DeltaProduct
The state-of-the-art method we build on (Siems et al., NeurIPS 2025). At every
step it updates the state by multiplying it by a matrix, and that matrix is
built as a **product of several simple pieces**. More pieces = more expressive
= more computation.

### Householder factor — one "piece"
```
H(β, k) = I − β · k kᵀ
```
Think of it as an operation that **picks one direction and only changes that
direction**, leaving everything perpendicular untouched.

### β (beta) — how strongly the piece acts
A number between 0 and 2 that sets *how much* the chosen direction is changed:

| β | effect |
|---|---|
| **0** | do nothing at all (the piece becomes invisible) |
| **1** | erase that direction |
| **2** | flip that direction — a **mirror reflection** |

β = 2 is the important one, because **a swap of two cups is exactly a mirror
reflection.** That is the whole reason this architecture can track cup
shuffling: chain enough reflections and you can build any rearrangement.

### k — the direction
A unit-length vector saying *which* direction the piece acts on. Produced by the
network from the current input.

### n_h — the "order" (also `num_householder`)
**How many pieces are used per input symbol.** This is the single most important
number in the project.

- In the original method, `n_h` is **one global number fixed by hand** before
  training and applied identically to every symbol.
- Our proposal is to make it **vary per symbol, and be learned**.

`fixed3` in our logs means n_h fixed at 3 for every symbol.

### head, head_dim
The model runs several independent copies of the state in parallel, called
heads, and combines them at the end. We use **12 heads, each with a
32-dimensional state**. More heads = more capacity, same idea.

### Representation (ρ)
A way of writing group operations as matrices so the model can actually apply
them. **A group can have several different representations, of different
sizes,** and the model is free to use whichever it likes. This freedom is the
source of the Part 4 result.

### rank(M − I)
"How many independent directions does this matrix actually change?" If a matrix
leaves most directions alone, this number is small. It decides the **minimum
number of pieces** needed, because each piece changes only one direction.

---

## Part 3 — Our additions

### Halting gate
A small network that looks at each input symbol and decides **how many of the
available pieces to actually use** for that symbol. Unused pieces get β = 0, so
they do nothing.

### Monotone halting
Once the gate decides to stop, it stays stopped — you cannot use piece 1, skip
piece 2, then use piece 3. This makes the count a single sensible number.
Borrowed from PonderNet / Adaptive Computation Time.

### Order / effective order (n_t)
How many pieces symbol *t* actually used after gating. Averaged over the data,
this is the model's real per-symbol cost.

### Oracle
A **control condition**, not a model. Instead of *learning* how many pieces each
symbol needs, we simply **tell it the correct number** from ground truth. This
measures the best a perfect gate could possibly do. Every adaptive result in the
project so far uses the oracle — **the gate has not yet been trained
end-to-end**, and nothing should be described as if it has.

### Budget loss
A penalty term that pushes the gate toward using fewer pieces. Written and
tested, **not yet connected to the training loop.**

### Path A vs Path B
- **Path A** (what we have): all pieces are still computed, unused ones are
  multiplied by zero. Correct results, **no actual speed saving**.
- **Path B** (not yet built): genuinely skip the unused pieces. This is what
  turns the measured saving into real wall-clock speed.

> Every "cost" number in our results is **required work per symbol**, not
> measured speed. At p = 0.1 the full-cost model took 761s and the adaptive one
> 756s — essentially identical, because Path B does not exist yet.

---

## Part 4 — The new idea (our contribution)

### D(g) — the demand
**The true minimum number of pieces an instruction requires:**

> D(g) = the smallest value of rank(ρ(g) − I), taken over every valid
> representation ρ of **the group generated by the whole alphabet**.

### Why this is not the same as ℓ
Everyone, including us until recently, assumed the cost of an instruction is
ℓ(g) — its transposition length. That is right for S₅ but **wrong in general**:

| group in play | a 5-cycle's ℓ | a 5-cycle's true demand D |
|---|---|---|
| S₅ (all 120 rearrangements) | 4 | **4** |
| A₅ (the even 60) | 4 | **2** |

The instruction is identical in both rows. What changed is the group the rest of
the alphabet forces the model to handle. A₅ can be represented in only **3
dimensions** (it is the rotation group of an icosahedron), and in 3 dimensions
every rotation is just two mirror reflections. S₅ has no such compact
representation.

### The consequence, in one sentence
**How much work an instruction costs is not a property of that instruction — it
is a property of the entire set of instructions the task uses.**

So adding *one swap* — the easiest possible instruction — to an alphabet of
5-cycles doubles the cost of every 5-cycle, because it turns A₅ into S₅.
**An easy instruction makes the task harder.**

### Where this helps us, and where it does not
| group | demand varies across instructions? | can adaptive computation help? |
|---|---|---|
| S₃, S₄, S₅ | yes (1, 2, 3, 4) | **yes** |
| A₄, A₅ | no (always 2) | **no** |

We record the negative case honestly: on A₄ and A₅ our mechanism cannot help at
all, and we can now predict that in advance instead of discovering it by running.

---

## Part 5 — Reading an experiment log

### Arm
One configuration being compared. `fixed1`…`fixed4` use that many pieces for
every symbol; `oracle` uses the correct number per symbol.

### Token accuracy
Fraction of individual positions answered correctly. Forgiving.

### Sequence accuracy
Fraction of sequences where **every single position** is correct. Very harsh —
one slip anywhere fails the whole sequence. With 128 positions, 99% per position
gives only 28% sequence accuracy.

### k — sequence length
How many instructions per example. We train at **k = 128**.

### Length extrapolation
Training at length 128 and testing at 512. Tests whether the model learned a
genuine rule or memorised the training length.

### steps, batch
`steps` = training iterations, `batch` = examples per iteration. Our standard
run is 20 000 steps × batch 128.

### loss
Cross-entropy. **4.79 means random guessing** (there are 120 possible answers).
**Below about 0.01 means solved.** A loss stuck near 4.6 means the model has
learned nothing.

### The noise floor — and why there are two of them

The GPU kernels are non-deterministic, so the same configuration with the same
seed does not give the same answer twice. **How different depends entirely on
which kind of cell you are on, and confusing the two has cost us real results.**

**On a cell that reliably solves** (e.g. `fixed3` at p=0.1), two runs at the same
seed gave 0.9696 and 0.9740. So:

> differences below **~0.005 token / ~0.05 sequence accuracy are noise.**

**On a bistable cell, that figure is meaningless.** `A5_c5:n_h=2` at the *same
seed* has produced 0.1177 and 0.9598 — a spread of **0.84 token accuracy**. These
cells sit on a knife edge: the run either escapes the loss plateau or it does not,
and non-determinism alone decides which.

> On a bistable cell, **do not report accuracy at all as a single number.**
> Report the **escape fraction** — how many runs of N reached the solution — and
> the sorted list. A mean over a bimodal distribution describes no run that
> actually happened.

**How to tell which you have:** run it 10 times. If every run lands in the same
place, the first floor applies. If runs split between ~0.999 and ~0.04, it is
bistable. As of 13 September, every cell we had called a "training failure"
turned out to be bistable — see `AUDIT.md` §5.

---

## Part 6 — Symbols at a glance

| symbol | meaning | typical value |
|---|---|---|
| `n` | number of items being shuffled | 5 |
| `Sₙ` | all rearrangements of n items | S₅ = 120 |
| `Aₙ` | the even half | A₅ = 60 |
| `g` | one element (one instruction) | — |
| `ℓ(g)` | fewest swaps to build g | 1 for a swap, 4 for a 5-cycle |
| `D(g)` | **true minimum pieces required** | 4 in S₅, 2 in A₅ |
| `⟨A⟩` | group generated by alphabet A | A₅ or S₅ |
| `p` | fraction of hard instructions | 0.05 – 1.0 |
| `E[ℓ]` | mean work per symbol | 1 + 3p |
| `n_h` | pieces per symbol (the "order") | 1 – 4 |
| `n_t` | pieces symbol t actually used | 0 – n_h |
| `β` | strength of one piece | 0 – 2 |
| `k` | direction of one piece | unit vector |
| `H(β,k)` | one piece: `I − β kkᵀ` | — |
| `K` | number of pieces in a product | — |
| `k` (lowercase, in configs) | sequence length | 128 |

> **Note the collision:** `k` means the *direction vector* in the mathematics and
> the *sequence length* in the command-line flags (`--k 128`). Context
> distinguishes them; we have not renamed either because both match their
> source (the paper, and the authors' code).

---

## Part 7 — Files

| file | what it does | needs GPU |
|---|---|---|
| `work/exp_minimal_order.py` | verifies the demand model D(g); builds A₅'s 3-D representation | no |
| `work/exp_beta_reachability.py` | proves the two limits on β and the piece count | no |
| `work/exp_group_closure.py` | **the decisive test**: does one extra easy symbol break it? | yes |
| `work/exp_matched_compute.py` | main comparison: fixed budgets vs adaptive | yes |
| `work/exp_error_forensics.py` | how and where the models fail | yes |
| `work/gen_heterogeneous.py` | builds the mixed easy/hard datasets | no |
| `src/gating.py` | the halting gate | no |
| `src/oracle_order.py` | supplies ground-truth order to the oracle | no |
| `PLAN.md` | full record, claims, evidence, plan | — |

Every experiment file has a self-test that runs against data whose answer is
known in advance. Run `python work/<file>.py --selftest` where available.
