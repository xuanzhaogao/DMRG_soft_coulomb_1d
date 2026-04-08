#!/usr/bin/env python3
"""
Fit SOE approximation of 1/sqrt(1+t^2) on [0, 40]:
  1) VARPRO (Variable Projection) to get high-accuracy multi-exponential fit
  2) TLBT (Time-Limited Balanced Truncation) to compress

Target: 6-digit accuracy (max abs error ~ 1e-6).
"""
import os
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import svd, eig, inv


# ================================================================
# Evaluation and error checking
# ================================================================

def eval_soe(t, s, w):
    """Evaluate Re[Σ w_k exp(-s_k t)]"""
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


# ================================================================
# Step 1: VARPRO fit — real exponents only for stability
# ================================================================

def varpro_fit(K, tmax=40.0, npts=2000):
    """
    Fit 1/sqrt(1+t^2) ≈ Σ_{k=1}^{K} w_k exp(-s_k t) on [0, tmax]
    using Variable Projection.

    s_k are real positive (enforced via log parameterization).
    w_k are real (solved by linear least squares for given s).
    """
    # Dense sampling, with more points near t=0 where kernel changes rapidly
    t1 = np.linspace(0, 2, npts // 2)
    t2 = np.linspace(2, tmax, npts // 2)
    t = np.unique(np.concatenate([t1, t2]))
    f = 1.0 / np.sqrt(1.0 + t**2)

    # Weight: emphasize small t (large kernel values) and moderate t
    weight = np.ones_like(t)

    def solve_weights(log_s):
        """For given s, solve for optimal w by linear least squares."""
        s = np.exp(log_s)
        # Build matrix A: A_ij = exp(-s_j * t_i) * weight_i
        A = np.exp(-np.outer(t, s)) * weight[:, None]
        b = f * weight
        w, res, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        return w

    def objective(log_s):
        """Residual norm for given s (w solved by lstsq)."""
        s = np.exp(log_s)
        A = np.exp(-np.outer(t, s)) * weight[:, None]
        b = f * weight
        w, res, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        residual = A @ w - b
        return 0.5 * np.sum(residual**2)

    # Initial guess: log-spaced exponents covering the range
    # Small s captures slow tail, large s captures cusp at t=0
    s_init = np.logspace(-5, 2, K)
    log_s_init = np.log(s_init)

    print(f"  VARPRO: fitting {K} real exponentials on [0, {tmax}]...")
    result = minimize(objective, log_s_init, method='L-BFGS-B',
                      options={'maxiter': 5000, 'ftol': 1e-30, 'gtol': 1e-15})

    s_opt = np.exp(result.x)
    w_opt = solve_weights(result.x)

    # Convert to complex arrays for compatibility
    s_out = s_opt.astype(complex)
    w_out = w_opt.astype(complex)

    max_err, mean_err = check_error(s_out, w_out, tmax)
    print(f"  VARPRO result: {K} terms, max err = {max_err:.3e}, mean err = {mean_err:.3e}")
    print(f"  s range: [{np.min(s_opt):.3e}, {np.max(s_opt):.3e}]")
    print(f"  max |w|: {np.max(np.abs(w_opt)):.3e}")

    return s_out, w_out, max_err


def multi_start_varpro(K, tmax=40.0, npts=2000, n_starts=5):
    """Run VARPRO with multiple random initializations."""
    best_s, best_w, best_err = None, None, 1e10

    for trial in range(n_starts):
        t1 = np.linspace(0, 2, npts // 2)
        t2 = np.linspace(2, tmax, npts // 2)
        t = np.unique(np.concatenate([t1, t2]))
        f = 1.0 / np.sqrt(1.0 + t**2)
        weight = np.ones_like(t)

        # Perturbed initial guess
        if trial == 0:
            s_init = np.logspace(-5, 2, K)
        elif trial == 1:
            s_init = np.logspace(-6, 2.5, K)
        elif trial == 2:
            s_init = np.logspace(-4, 1.5, K)
        else:
            rng = np.random.RandomState(trial)
            s_init = np.sort(np.exp(rng.uniform(-6, 2.5, K)))

        log_s_init = np.log(s_init)

        def solve_weights(log_s):
            s = np.exp(log_s)
            A = np.exp(-np.outer(t, s)) * weight[:, None]
            b = f * weight
            w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            return w

        def objective(log_s):
            s = np.exp(log_s)
            A = np.exp(-np.outer(t, s)) * weight[:, None]
            b = f * weight
            w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            residual = A @ w - b
            return 0.5 * np.sum(residual**2)

        result = minimize(objective, log_s_init, method='L-BFGS-B',
                          options={'maxiter': 10000, 'ftol': 1e-30, 'gtol': 1e-15})

        s_opt = np.exp(result.x).astype(complex)
        w_opt = solve_weights(result.x).astype(complex)
        max_err, mean_err = check_error(s_opt, w_opt, tmax)

        print(f"    trial {trial}: max err = {max_err:.3e}")
        if max_err < best_err:
            best_s, best_w, best_err = s_opt, w_opt, max_err

    return best_s, best_w, best_err


# ================================================================
# Step 2: TLBT model reduction
# ================================================================

def tlbt_reduce(s, w, p, T=40.0):
    """
    TLBT for SOE: f(t) = Σ w_k exp(-s_k t).
    State-space: A = diag(-s), B = w (column), C = 1 (row).
    """
    n = len(s)
    A = np.diag(-s)
    B = w.reshape(-1, 1)
    C = np.ones((1, n))
    EA = np.exp(-T * s)

    SS = s.reshape(-1, 1) + s.conj().reshape(1, -1)
    factor = (1.0 - np.outer(EA, EA.conj())) / SS
    P = (B @ B.conj().T) * factor

    SS_obs = s.conj().reshape(-1, 1) + s.reshape(1, -1)
    factor_obs = (1.0 - np.outer(EA.conj(), EA)) / SS_obs
    Q = (C.conj().T @ C) * factor_obs

    P = 0.5 * (P + P.conj().T)
    Q = 0.5 * (Q + Q.conj().T)

    eigP, VP = np.linalg.eigh(P)
    eigQ, VQ = np.linalg.eigh(Q)

    tol_eig = 1e-15 * max(np.max(np.abs(eigP)), np.max(np.abs(eigQ)))
    maskP = eigP > tol_eig
    maskQ = eigQ > tol_eig

    RP = VP[:, maskP] @ np.diag(np.sqrt(eigP[maskP]))
    RQ = VQ[:, maskQ] @ np.diag(np.sqrt(eigQ[maskQ]))

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
    B_red = np.linalg.solve(V, Bd)
    C_red = Cd @ V
    w_red = (B_red.flatten()) * (C_red.flatten())

    return s_red, w_red, Sigma


# ================================================================
# Save
# ================================================================

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
        f.write(f"# {'All decaying (Re(s) > 0)' if all_decay else 'Non-decaying modes present'}\n")
        f.write(f"# Max abs error:  {max_err:.2e}\n")
        f.write(f"# Mean abs error: {mean_err:.2e}\n")
        f.write(f"#\n")
        f.write(f"# {n_real} real terms + {n_cpx} conjugate pairs = {len(s)} terms total\n")
        f.write(f"#\n")
        f.write(f"# Columns: Re(s)  Im(s)  Re(w)  Im(w)\n")
        for sk, wk in zip(s, w):
            f.write(f"  {sk.real:+24.16e}  {sk.imag:+24.16e}  {wk.real:+24.16e}  {wk.imag:+24.16e}\n")

    print(f"\n  Saved: {filename}")
    print(f"  {len(s)} terms ({n_real} real + {n_cpx} pairs)")
    print(f"  max err = {max_err:.3e}, mean err = {mean_err:.3e}")
    print(f"  all decaying: {all_decay}")


# ================================================================
# Main
# ================================================================

def main():
    T = 40.0
    target_err = 1e-6

    # Step 1: Find optimal number of VARPRO terms
    print("=" * 60)
    print("Step 1: VARPRO fit (real exponents, all decaying)")
    print("=" * 60)

    # Scan number of terms
    varpro_results = {}
    for K in [20, 25, 30, 35, 40, 50, 60]:
        print(f"\n--- K = {K} ---")
        s, w, max_err = multi_start_varpro(K, tmax=T, npts=3000, n_starts=8)
        varpro_results[K] = (s, w, max_err)
        if max_err < target_err * 0.1:
            print(f"  Sufficient accuracy reached at K={K}")
            break

    # Pick the smallest K that gives good accuracy for TLBT input
    # We want max_err < target / 10 to have margin for TLBT compression loss
    best_K = None
    for K in sorted(varpro_results.keys()):
        s, w, max_err = varpro_results[K]
        if max_err < target_err:
            best_K = K
            break
    if best_K is None:
        # Just use the largest K
        best_K = max(varpro_results.keys())

    s_full, w_full, err_full = varpro_results[best_K]
    print(f"\nUsing K={best_K}: {len(s_full)} terms, max err = {err_full:.3e}")

    save_soe("sog/soe_full_varpro.txt", s_full, w_full,
             f"Full VARPRO fit ({best_K} real exponentials)", T)

    # Step 2: TLBT reduction
    print("\n" + "=" * 60)
    print("Step 2: TLBT model reduction")
    print("=" * 60)

    results = []
    for p in range(10, min(len(s_full) - 1, 50)):
        try:
            s_red, w_red, sigma = tlbt_reduce(s_full, w_full, p, T)
            max_err, mean_err = check_error(s_red, w_red, T)
            all_decay = np.all(s_red.real > -1e-10)
            tag = "" if all_decay else " [non-decaying]"
            results.append((p, max_err, mean_err, all_decay, s_red, w_red))
            print(f"  p={p:2d}: max err = {max_err:.3e}, mean = {mean_err:.3e}{tag}")
            if max_err < target_err * 0.01:
                break  # well below target, no need to go higher
        except Exception as e:
            print(f"  p={p:2d}: FAILED ({e})")

    # Save best results
    print("\n" + "=" * 60)
    print("Step 3: Select and save")
    print("=" * 60)

    # Smallest p meeting target with all-decaying constraint
    for p, max_err, mean_err, all_decay, s_red, w_red in results:
        if max_err < target_err and all_decay:
            save_soe(f"sog/soe_{p}term_decaying.txt", s_red, w_red,
                     f"{p}-term TLBT from {best_K}-term VARPRO (all decaying)", T)
            break
    else:
        print("  No all-decaying result met target")

    # Smallest p meeting target (no constraint)
    for p, max_err, mean_err, all_decay, s_red, w_red in results:
        if max_err < target_err:
            if not all_decay:
                save_soe(f"sog/soe_{p}term_noconstraint.txt", s_red, w_red,
                         f"{p}-term TLBT from {best_K}-term VARPRO (no decay constraint)", T)
            break


if __name__ == "__main__":
    main()
