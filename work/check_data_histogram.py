"""
Week 1, Day 4: confirm the data generator samples UNIFORMLY over the whole group.

Reading generate_data.py says it does (allowed_indices = range(num_elements),
then random.choices). This verifies it empirically on the generated CSV, which
is what fixes the efficiency ceiling from Part 4 of the plan:

    ceiling = (H_n - 1)/(n - 1)      S5 -> 32.1%

We deliberately do NOT try to map token indices to specific permutations (the
element ordering in abstract_algebra need not match itertools.permutations).
Uniformity over all |G| elements is enough: the transposition-length histogram
then follows from census.py.

    python work/check_data_histogram.py state_tracking/data/S5=128.csv
"""
import sys, collections, math, csv, pathlib

def main(path):
    p = pathlib.Path(path)
    if not p.exists():
        sys.exit(f"not found: {p}\nGenerate it first with src/generate_data.py")

    counts = collections.Counter()
    rows = 0
    with p.open() as f:
        r = csv.reader(f)
        header = next(r)
        print(f"columns: {header}")
        icol = 0
        for row in r:
            rows += 1
            counts.update(row[icol].split())
            if rows >= 20000:
                break

    n_elem = len(counts)
    total = sum(counts.values())
    freqs = [c / total for c in counts.values()]
    expected = 1.0 / n_elem
    maxdev = max(abs(f - expected) for f in freqs) / expected

    print(f"rows scanned        : {rows}")
    print(f"distinct elements   : {n_elem}")
    print(f"expected freq each  : {expected:.5f}")
    print(f"max relative dev    : {maxdev*100:.2f}%")

    for n in range(3, 9):
        if n_elem == math.factorial(n):
            H = sum(1.0/i for i in range(1, n+1))
            print(f"\n-> matches |S{n}| = {n_elem}")
            print(f"   E[transposition length] = n - H_n = {n - H:.4f}")
            print(f"   max order needed        = {n-1}")
            print(f"   EFFICIENCY CEILING      = {(H-1)/(n-1)*100:.1f}%")
            break
        if n_elem == math.factorial(n)//2:
            print(f"\n-> matches |A{n}| = {n_elem} (alternating group)")
            break
    else:
        print(f"\n-> |G| = {n_elem} does not match a plain S_n or A_n; "
              f"check whether a 'limit_to' / 'only_swaps' variant was used.")

    verdict = "UNIFORM (as expected)" if maxdev < 0.15 else "NOT uniform — investigate"
    print(f"\nsampling: {verdict}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "state_tracking/data/S5=128.csv")
