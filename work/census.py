from itertools import permutations
from fractions import Fraction

def cycles(p):
    n=len(p); seen=[False]*n; c=0
    for i in range(n):
        if not seen[i]:
            c+=1; j=i
            while not seen[j]: seen[j]=True; j=p[j]
    return c

def parity(p):  # even permutation?
    return (len(p)-cycles(p))%2==0

def stats(elems,n,name):
    L=[n-cycles(p) for p in elems]
    N=len(L); mx=max(L); E=Fraction(sum(L),N)
    dist={}
    for l in L: dist[l]=dist.get(l,0)+1
    waste=1-E/mx
    print(f"{name:6s} |G|={N:4d}  max_l={mx}  E[l]={float(E):.4f}  "
          f"ceiling={float(waste)*100:5.1f}%  dist={dict(sorted(dist.items()))}")
    return float(E),mx

for n in range(3,9):
    S=list(permutations(range(n)))
    stats(S,n,f"S{n}")
    if n<=6:
        A=[p for p in S if parity(p)]
        stats(A,n,f"A{n}")

print("\nclosed form check:  E[l] = n - H_n ,  ceiling = (H_n - 1)/(n - 1)")
H=0.0
for n in range(1,11):
    H+=1/n
    if n>=3:
        print(f"  n={n}: E[l]={n-H:.4f}  ceiling={(H-1)/(n-1)*100:5.1f}%")
