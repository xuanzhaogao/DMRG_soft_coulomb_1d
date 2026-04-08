#!/usr/bin/env python3
"""
Fit SOE 1/sqrt(1+t^2) on [0, 40] with WELL-CONDITIONED coefficients.

The key constraint: max|w_k| must be small (< ~10) for the MPO to work
in double precision DMRG. We use L2 regularization on weights and
bounded exponents to prevent clustering.
"""
import numpy as np
from scipy.optimize import minimize, differential_evolution


def eval_soe(t, s, w):
    result = np.zeros_like(t)
    for sk, wk in zip(s, w):
        result += wk * np.exp(-sk * t)
    return result


def check_error(s, w, tmax=40.0, npts=50000):
    t = np.linspace(0, tmax, npts)
    exact = 1.0 / np.sqrt(1.0 + t**2)
    approx = eval_soe(t, s, w)
    err = np.abs(exact - approx)
    return np.max(err), np.mean(err)


def fit_conditioned(K, tmax=40.0, npts=4000, max_w=5.0, seed=0):
    """
    Fit using bounded VARPRO: exponents must stay well-separated,
    weights penalized if > max_w.
    """
    t1 = np.linspace(0, 2, npts // 2)
    t2 = np.linspace(2, tmax, npts // 2)
    t = np.unique(np.concatenate([t1, t2]))
    f = 1.0 / np.sqrt(1.0 + t**2)

    def solve_weights(s):
        A = np.exp(-np.outer(t, s))
        w, _, _, _ = np.linalg.lstsq(A, f, rcond=None)
        return w

    def objective(log_s):
        s = np.exp(log_s)
        A = np.exp(-np.outer(t, s))
        w, _, _, _ = np.linalg.lstsq(A, f, rcond=None)
        residual = A @ w - f
        fit_err = np.sum(residual**2)

        # Penalty for large weights
        w_penalty = np.sum(np.maximum(np.abs(w) - max_w, 0)**2) * 1e4

        # Penalty for exponent clustering (minimum separation in log space)
        log_s_sorted = np.sort(log_s)
        gaps = np.diff(log_s_sorted)
        min_gap = np.log(10) * 0.15  # at least 10^0.15 ≈ 1.4x separation
        gap_penalty = np.sum(np.maximum(min_gap - gaps, 0)**2) * 1e6

        return fit_err + w_penalty + gap_penalty

    # Initial guess: well-separated log-spaced exponents
    rng = np.random.RandomState(seed)
    if seed == 0:
        s_init = np.logspace(np.log10(1e-4), np.log10(200), K)
    elif seed == 1:
        s_init = np.logspace(np.log10(5e-5), np.log10(150), K)
    elif seed == 2:
        s_init = np.logspace(np.log10(2e-4), np.log10(250), K)
    else:
        lo, hi = rng.uniform(-5, -3), rng.uniform(1.5, 2.5)
        s_init = np.logspace(lo, hi, K)

    log_s = np.log(s_init)

    # Bounds to keep exponents positive and in reasonable range
    bounds = [(-12, 6)] * K

    result = minimize(objective, log_s, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 20000, 'ftol': 1e-30, 'gtol': 1e-15})

    s_opt = np.exp(result.x)
    w_opt = solve_weights(s_opt)

    # Sort by exponent
    order = np.argsort(s_opt)
    s_opt = s_opt[order]
    w_opt = w_opt[order]

    return s_opt, w_opt


def save_soe(filename, s, w, label, tmax=40.0):
    max_err, mean_err = check_error(s, w, tmax)
    with open(filename, 'w') as f:
        f.write(f"# Sum-of-Exponentials (SOE) approximation of the soft-Coulomb kernel\n")
        f.write(f"# 1/sqrt(1+t^2) ≈ Σ_{{k=1}}^{{{len(s)}}} w_k * exp(-s_k * t)\n")
        f.write(f"#\n")
        f.write(f"# {label}\n")
        f.write(f"# Valid on t ∈ [0, {tmax:.0f}]\n")
        f.write(f"# All real, all decaying (s_k > 0)\n")
        f.write(f"# Max abs error:  {max_err:.2e}\n")
        f.write(f"# Mean abs error: {mean_err:.2e}\n")
        f.write(f"# Max |w_k|:      {np.max(np.abs(w)):.4f}\n")
        f.write(f"#\n")
        f.write(f"# {len(s)} real terms\n")
        f.write(f"#\n")
        f.write(f"# Columns: s_k  w_k\n")
        for sk, wk in zip(s, w):
            f.write(f"  {sk:+24.16e}  {wk:+24.16e}\n")
    print(f"  Saved: {filename}")


def main():
    T = 40.0
    target = 1e-6
    n_trials = 15

    print("=" * 60)
    print("Well-conditioned SOE fit for MPO construction")
    print("=" * 60)

    for K in [20, 25, 30, 35, 40, 50]:
        print(f"\n--- K = {K} ---")
        best_s, best_w, best_err = None, None, 1e10
        best_maxw = 1e10

        for trial in range(n_trials):
            for max_w in [2.0, 5.0, 10.0]:
                s, w = fit_conditioned(K, tmax=T, max_w=max_w, seed=trial)
                me, mn = check_error(s, w, T)
                mw = np.max(np.abs(w))
                if me < best_err and mw < 50:
                    best_s, best_w, best_err = s, w, me
                    best_maxw = mw
                    best_trial = trial
                    best_mw_param = max_w

        if best_s is not None:
            print(f"  Best: max err = {best_err:.3e}, max|w| = {best_maxw:.2f} "
                  f"(trial={best_trial}, max_w={best_mw_param})")
            print(f"  s range: [{np.min(best_s):.3e}, {np.max(best_s):.3e}]")

            if best_err < target:
                print(f"  >>> TARGET MET!")
                save_soe(f"sog/soe_{K}term_conditioned.txt", best_s, best_w,
                         f"{K}-term conditioned VARPRO (max|w|={best_maxw:.2f})", T)
                break
        else:
            print(f"  No good result found")


if __name__ == "__main__":
    main()
