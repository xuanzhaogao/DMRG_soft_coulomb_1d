#!/usr/bin/env python3
"""
Final SOE fitting for 1/sqrt(1+t^2) on [0, 40].
  1) VARPRO to get high-accuracy fits at various K
  2) TLBT to compress
  3) Save both all-decaying and compact variants
"""
import os
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import svd, eig, inv


def eval_soe(t, s, w):
    result = np.zeros_like(t, dtype=complex)
    for sk, wk in zip(s, w):
        result += wk * np.exp(-sk * t)
    return result.real


def check_error(s, w, tmax=40.0, npts=50000):
    t = np.linspace(0, tmax, npts)
    exact = 1.0 / np.sqrt(1.0 + t**2)
    approx = eval_soe(t, s, w)
    err = np.abs(exact - approx)
    return np.max(err), np.mean(err)


def varpro_single(K, tmax=40.0, npts=3000, seed=0):
    """Single VARPRO trial with given seed."""
    t1 = np.linspace(0, 2, npts // 2)
    t2 = np.linspace(2, tmax, npts // 2)
    t = np.unique(np.concatenate([t1, t2]))
    f = 1.0 / np.sqrt(1.0 + t**2)

    rng = np.random.RandomState(seed)
    if seed == 0:
        s_init = np.logspace(-5, 2, K)
    elif seed == 1:
        s_init = np.logspace(-6, 2.5, K)
    elif seed == 2:
        s_init = np.logspace(-4, 1.5, K)
    elif seed == 4:
        s_init = np.logspace(-7, 3, K)
    else:
        s_init = np.sort(np.exp(rng.uniform(-7, 3, K)))

    log_s = np.log(s_init)

    def objective(log_s):
        s = np.exp(log_s)
        A = np.exp(-np.outer(t, s))
        w, _, _, _ = np.linalg.lstsq(A, f, rcond=None)
        return 0.5 * np.sum((A @ w - f)**2)

    result = minimize(objective, log_s, method='L-BFGS-B',
                      options={'maxiter': 10000, 'ftol': 1e-30, 'gtol': 1e-15})
    s_opt = np.exp(result.x)
    A = np.exp(-np.outer(t, s_opt))
    w_opt, _, _, _ = np.linalg.lstsq(A, f, rcond=None)
    return s_opt.astype(complex), w_opt.astype(complex)


def tlbt_reduce(s, w, p, T=40.0):
    n = len(s)
    A = np.diag(-s)
    B = w.reshape(-1, 1)
    C = np.ones((1, n))
    EA = np.exp(-T * s)

    SS = s.reshape(-1, 1) + s.conj().reshape(1, -1)
    P = (B @ B.conj().T) * ((1.0 - np.outer(EA, EA.conj())) / SS)
    SS_obs = s.conj().reshape(-1, 1) + s.reshape(1, -1)
    Q = (C.conj().T @ C) * ((1.0 - np.outer(EA.conj(), EA)) / SS_obs)

    P = 0.5 * (P + P.conj().T)
    Q = 0.5 * (Q + Q.conj().T)

    eigP, VP = np.linalg.eigh(P)
    eigQ, VQ = np.linalg.eigh(Q)

    tol = 1e-15 * max(np.max(np.abs(eigP)), np.max(np.abs(eigQ)))
    RP = VP[:, eigP > tol] @ np.diag(np.sqrt(eigP[eigP > tol]))
    RQ = VQ[:, eigQ > tol] @ np.diag(np.sqrt(eigQ[eigQ > tol]))

    LL = RP.conj().T @ RQ
    U, Sigma, Vh = svd(LL)
    if p > len(Sigma):
        raise ValueError(f"p={p} > rank ({len(Sigma)})")

    Sigma_inv_half = np.diag(Sigma[:p] ** -0.5)
    Trans = RP @ U[:, :p] @ Sigma_inv_half
    invT = Sigma_inv_half @ Vh[:p, :] @ RQ.conj().T

    Ad = invT @ A @ Trans
    Bd = invT @ B
    Cd = C @ Trans

    d, V = eig(Ad)
    s_red = -d
    w_red = (np.linalg.solve(V, Bd).flatten()) * ((Cd @ V).flatten())
    return s_red, w_red, Sigma


def save_soe(filename, s, w, label, tmax=40.0):
    max_err, mean_err = check_error(s, w, tmax)
    n_real = int(np.sum(np.abs(s.imag) < 1e-12))
    n_cpx = (len(s) - n_real) // 2
    all_decay = np.all(s.real > -1e-12)

    with open(filename, 'w') as f:
        f.write(f"# Sum-of-Exponentials (SOE) approximation of the soft-Coulomb kernel\n")
        f.write(f"# 1/sqrt(1+t^2) ≈ Re[ Σ_{{k=1}}^{{{len(s)}}} w_k * exp(-s_k * t) ]\n")
        f.write(f"#\n")
        f.write(f"# {label}\n")
        f.write(f"# Valid on t ∈ [0, {tmax:.0f}]\n")
        f.write(f"# {'All decaying (Re(s) > 0)' if all_decay else 'Non-decaying modes present (mildly bounded)'}\n")
        f.write(f"# Max abs error:  {max_err:.2e}\n")
        f.write(f"# Mean abs error: {mean_err:.2e}\n")
        f.write(f"#\n")
        f.write(f"# {n_real} real terms + {n_cpx} conjugate pairs = {len(s)} terms total\n")
        f.write(f"#\n")
        f.write(f"# Columns: Re(s)  Im(s)  Re(w)  Im(w)\n")
        for sk, wk in zip(s, w):
            f.write(f"  {sk.real:+24.16e}  {sk.imag:+24.16e}  {wk.real:+24.16e}  {wk.imag:+24.16e}\n")

    print(f"  Saved: {filename}")
    print(f"  {len(s)} terms ({n_real} real + {n_cpx} pairs), all_decay={all_decay}")
    print(f"  max err = {max_err:.3e}, mean err = {mean_err:.3e}")


def main():
    T = 40.0
    target = 1e-6
    n_trials = 12

    # ============================================================
    # Step 1: VARPRO fits at various K
    # ============================================================
    print("=" * 60)
    print("Step 1: VARPRO fitting")
    print("=" * 60)

    all_fits = {}
    for K in [20, 25, 30, 35, 40]:
        print(f"\n  K={K}:")
        best_s, best_w, best_err = None, None, 1e10
        for trial in range(n_trials):
            s, w = varpro_single(K, tmax=T, seed=trial)
            me, _ = check_error(s, w, T)
            if me < best_err:
                best_s, best_w, best_err = s, w, me
            print(f"    trial {trial:2d}: max err = {me:.3e}" + (" *" if me == best_err else ""))
        all_fits[K] = (best_s, best_w, best_err)
        print(f"  Best K={K}: max err = {best_err:.3e}")

    # ============================================================
    # Step 2: Save best all-decaying VARPRO fit that meets target
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 2: Select best VARPRO fit (all-decaying)")
    print("=" * 60)

    for K in sorted(all_fits.keys()):
        s, w, err = all_fits[K]
        if err < target:
            save_soe(f"sog/soe_{K}term_varpro.txt", s, w,
                     f"{K}-term VARPRO fit (all real, all decaying)", T)
            break

    # ============================================================
    # Step 3: TLBT compression from each VARPRO fit
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 3: TLBT compression")
    print("=" * 60)

    best_compact = {}  # p -> (max_err, all_decay, s, w, source_K)

    for K in sorted(all_fits.keys()):
        s_full, w_full, err_full = all_fits[K]
        if err_full > 1e-2:
            continue
        print(f"\n  From K={K} (err={err_full:.3e}):")

        for p in range(10, min(len(s_full), 40)):
            try:
                s_red, w_red, sigma = tlbt_reduce(s_full, w_full, p, T)
                me, mn = check_error(s_red, w_red, T)
                ad = np.all(s_red.real > -1e-10)
                tag = "" if ad else " [nd]"
                print(f"    p={p:2d}: max err={me:.3e}{tag}")

                if p not in best_compact or me < best_compact[p][0]:
                    best_compact[p] = (me, ad, s_red, w_red, K)
            except:
                pass

    # ============================================================
    # Step 4: Save best results
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 4: Save results")
    print("=" * 60)

    # Smallest p meeting target (any)
    print("\n--- Compact (no constraint) ---")
    for p in sorted(best_compact.keys()):
        me, ad, s_red, w_red, src = best_compact[p]
        if me < target:
            save_soe(f"sog/soe_{p}term_compact.txt", s_red, w_red,
                     f"{p}-term TLBT from K={src} VARPRO", T)
            break
    else:
        # Show best
        p = min(best_compact, key=lambda p: best_compact[p][0])
        me, ad, s_red, w_red, src = best_compact[p]
        save_soe(f"sog/soe_{p}term_compact.txt", s_red, w_red,
                 f"{p}-term TLBT from K={src} (best found, target not met)", T)

    # Smallest p meeting target (all decaying)
    print("\n--- Compact (all decaying) ---")
    for p in sorted(best_compact.keys()):
        me, ad, s_red, w_red, src = best_compact[p]
        if me < target and ad:
            save_soe(f"sog/soe_{p}term_compact_decay.txt", s_red, w_red,
                     f"{p}-term TLBT from K={src} (all decaying)", T)
            break
    else:
        print("  No all-decaying compact result met target")

    # Summary table
    print("\n" + "=" * 60)
    print("Summary of all TLBT results")
    print("=" * 60)
    print(f"{'p':>3s}  {'max_err':>10s}  {'src_K':>5s}  {'decay':>6s}")
    for p in sorted(best_compact.keys()):
        me, ad, _, _, src = best_compact[p]
        print(f"{p:3d}  {me:10.3e}  {src:5d}  {'yes' if ad else 'no':>6s}")


if __name__ == "__main__":
    main()
