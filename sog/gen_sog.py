import numpy as np
import os, sys

# Redirect C-level stdout to /dev/null during vpmr calls
from pyvpmr import vpmr

results = []
for n, c in [(10,4), (10,10), (15,6), (20,4), (30,10)]:
    # Redirect stdout at fd level
    old_stdout = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    try:
        m, s = vpmr(n=n, k='1/sqrt(1+t^2)', e=1e-4, c=c)
    finally:
        os.dup2(old_stdout, 1)
        os.close(devnull)
        os.close(old_stdout)

    max_wt = np.max(np.abs(m))
    t_test = np.linspace(0, 40, 1000)
    exact = 1.0 / np.sqrt(1 + t_test**2)
    approx = np.sum([mi * np.exp(-si * t_test) for mi, si in zip(m, s)], axis=0)
    err = np.max(np.abs(exact - approx.real))
    results.append((n, c, len(m), max_wt, err))

print("\nSOG parameter search results:")
print(f"{'n':>4s} {'c':>3s} {'terms':>6s} {'max_wt':>10s} {'max_err':>10s}")
for n, c, nt, mw, er in results:
    print(f"{n:4d} {c:3d} {nt:6d} {mw:10.1f} {er:10.4f}")
