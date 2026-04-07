# DMRG Reference Calculations for 1D Soft-Coulomb Systems

Reproducing and extending the benchmark results from Wu et al. (arXiv:2603.23897) using ITensor.jl.

## Physical Setup

Hamiltonian (Born-Oppenheimer, single atom fixed at origin):

$$H = \sum_{i=1}^N \left[ -\frac{1}{2}\nabla^2_{r_i} - z\, v(|r_i|) \right] + \sum_{i<j} v(r_{ij}), \qquad v(u) = \frac{1}{\sqrt{1+u^2}}$$

### Systems and Parameters

| System | $N$ | $z$ | $r_c$ (a.u.) | $N_\uparrow$ | $N_\downarrow$ | Target $E_0$ (Wu et al.) |
|--------|-----|-----|--------------|--------------|----------------|--------------------------|
| H      | 1   | 1   | 15           | 1            | 0              | -0.6697771382138         |
| He     | 2   | 2   | 13           | 1            | 1              | -2.2382578241080         |
| Li     | 3   | 3   | 20           | 2            | 1              | -4.210531647613249†      |
| Be     | 4   | 4   | 20           | 2            | 2              | -6.78507795170852†       |
| B      | 5   | 5   | 20           | 3            | 2              | -9.81971246294425†       |
| C      | 6   | 6   | 20           | 3            | 3              | -13.33154314459545†      |
| N      | 7   | 7   | 20           | 4            | 3              | -17.29164709368697†      |
| O      | 8   | 8   | 20           | 4            | 4              | -21.69957056180725†      |

† Extrapolated by Wu et al. from SOG-TNN convergence — not independently verified. Primary motivation for this calculation.

H and He values are from SG-CI (ref [59] in Wu et al.) and can be used as validation targets.

---

## Discretization

Uniform real-space grid on $[-r_c, r_c]$ with $L$ sites, spacing $\delta = 2r_c / L$.

**Kinetic energy** (second-order finite difference):
$$-\frac{1}{2}\nabla^2 \approx \frac{1}{\delta^2}\sum_i c^\dagger_i c_i - \frac{1}{2\delta^2}\sum_i \left(c^\dagger_{i+1}c_i + \text{h.c.}\right)$$

**Potential energy**: evaluated at grid points $r_i = -r_c + (i-1)\delta$.

Each site has four states: $\{|\text{Emp}\rangle, |\uparrow\rangle, |\downarrow\rangle, |\uparrow\downarrow\rangle\}$.

### Grid Convergence Targets

| $r_c$ | Suggested $L$ | $\delta$ (a.u.) |
|-------|---------------|-----------------|
| 13    | 260           | 0.10            |
| 15    | 300           | 0.10            |
| 20    | 400           | 0.10            |

Start with $\delta = 0.1$, verify by halving to $\delta = 0.05$. Energy should converge as $O(\delta^2)$.

---

## Long-Range Interaction: SOG Decomposition

Direct MPO construction for $v(|r_i - r_j|)$ gives MPO bond dimension $O(L)$, which is prohibitive.
Instead, use sum-of-Gaussians (SOG) decomposition of the soft-Coulomb kernel:

$$v(u) \approx \sum_{\ell=1}^{L_{\text{SOG}}} \tilde{\omega}_\ell\, e^{-\tilde{s}_\ell u^2}$$

Each Gaussian term $e^{-\tilde{s}_\ell(r_i - r_j)^2} = e^{-\tilde{s}_\ell r_i^2} e^{2\tilde{s}_\ell r_i r_j} e^{-\tilde{s}_\ell r_j^2}$ yields a rank-1 MPO contribution.
Total MPO bond dimension scales as $O(L_{\text{SOG}})$.

From Wu et al. Table I: with WBT compression at $\epsilon = 10^{-12}$, $L_{\text{SOG}} = 26$–$28$ depending on $r_c$.
Use the BSA parameters $\sigma=1$, $b=1.1878$ from Wu et al. as starting point, then apply WBT.

For the Gaussian $e^{-s(r_i - r_j)^2}$, the MPO operator string between sites $i$ and $j$ is:

$$e^{-s r_i^2} \left(\prod_{k=i+1}^{j-1} e^{0}\right) e^{-s r_j^2} \cdot e^{2s r_i r_j}$$

This requires careful handling of the cross term $e^{2s r_i r_j}$; expand as a finite sum using Chebyshev polynomials on $[-r_c, r_c]$ to obtain a sum of separable terms.

---

## ITensor.jl Implementation

### Dependencies

```julia
using ITensor
using ITensors.HDF5  # for saving results
```

### Site Indices with QN Conservation

```julia
function make_sites(L, N, Nup)
    # conserve_qns enforces fixed N and Sz = Nup - Ndn
    sites = siteinds("Electron", L; conserve_qns=true)
    return sites
end
```

### Hamiltonian Construction

```julia
function build_hamiltonian(L, δ, rc, z, sog_weights, sog_exponents)
    sites = siteinds("Electron", L; conserve_qns=true)
    os = OpSum()

    # --- Kinetic energy ---
    for i in 1:L
        os += 1/δ^2,       "Nup", i
        os += 1/δ^2,       "Ndn", i
    end
    for i in 1:L-1
        os += -1/(2δ^2),   "Cdagup", i, "Cup",   i+1
        os += -1/(2δ^2),   "Cup",    i, "Cdagup", i+1
        os += -1/(2δ^2),   "Cdagdn", i, "Cdn",   i+1
        os += -1/(2δ^2),   "Cdn",    i, "Cdagdn", i+1
    end

    # --- Electron-nucleus potential ---
    for i in 1:L
        ri = -rc + (i-1)*δ
        v_en = -z / sqrt(1 + ri^2)
        os += v_en, "Nup", i
        os += v_en, "Ndn", i
    end

    # --- Electron-electron interaction via SOG ---
    # For each Gaussian term: v_ee(r_i, r_j) ≈ Σ_ℓ ω_ℓ exp(-s_ℓ (r_i-r_j)^2)
    # Expand exp(-s(r_i-r_j)^2) = exp(-s*r_i^2) * exp(-s*r_j^2) * exp(2s*r_i*r_j)
    # Further expand exp(2s*r_i*r_j) ≈ Σ_m (2s*r_i*r_j)^m / m! up to order M_max
    # This gives separable terms amenable to MPO construction.
    #
    # For each SOG term ℓ and Taylor order m:
    for (ω, s) in zip(sog_weights, sog_exponents)
        for i in 1:L, j in i+1:L
            ri = -rc + (i-1)*δ
            rj = -rc + (j-1)*δ
            v_ij = ω * exp(-s*(ri-rj)^2)
            # spin-up / spin-up
            os += v_ij, "Nup", i, "Nup", j
            # spin-dn / spin-dn
            os += v_ij, "Ndn", i, "Ndn", j
            # spin-up / spin-dn (both orderings)
            os += v_ij, "Nup", i, "Ndn", j
            os += v_ij, "Ndn", i, "Nup", j
        end
    end

    H = MPO(os, sites)
    return H, sites
end
```

> **Note**: The double loop over $(i,j)$ above is correct but naive — it constructs a dense MPO. For production runs, implement the SOG-MPO directly using ITensor's `add` on rank-1 MPO terms to keep bond dimension $O(L_\text{SOG})$.

### Initial State

```julia
function initial_state(sites, L, N, Nup)
    Ndn = N - Nup
    state = fill("Emp", L)
    # spread electrons evenly to avoid bias
    up_positions = round.(Int, LinRange(1, L, Nup))
    dn_positions = round.(Int, LinRange(2, L-1, Ndn))
    for i in up_positions state[i] = "Up" end
    for i in dn_positions
        state[i] = (state[i] == "Up") ? "UpDn" : "Dn"
    end
    return MPS(sites, state)
end
```

### DMRG Sweeps

```julia
function run_dmrg(H, psi0; max_bond=1000, cutoff=1e-12, nsweeps=30)
    sweeps = Sweeps(nsweeps)
    setmaxdim!(sweeps,  50, 100, 200, 400, 600, 800, 1000,
               fill(max_bond, nsweeps-7)...)
    setcutoff!(sweeps,  fill(cutoff, nsweeps)...)
    # noise helps escape local minima in early sweeps
    setnoise!(sweeps,   1e-5, 1e-6, 1e-7, 1e-8,
               fill(0.0, nsweeps-4)...)

    energy, psi = dmrg(H, psi0, sweeps; outputlevel=1)
    return energy, psi
end
```

### Convergence Check

```julia
function check_convergence(H, psi)
    # Truncation error from last sweep is printed by dmrg()
    # Also check variance: <H^2> - <H>^2 should be < 1e-10
    E  = inner(psi', H, psi)
    E2 = inner(psi', H, H, psi)  # requires H^2 MPO or use apply
    variance = E2 - E^2
    @info "Energy = $E,  Variance = $variance"
    return E, variance
end
```

---

## Convergence Protocol

### Step 1: Validate on H and He

Compare against Wu et al. reference values:
- H: $E_0 = -0.6697771382138$
- He: $E_0 = -2.2382578241080$

Vary $L \in \{100, 200, 400\}$ and $\chi \in \{100, 200, 500\}$.
Accept when $|E_\text{DMRG} - E_\text{ref}| < 10^{-8}$.

### Step 2: Grid Convergence for Li–O

For each system, compute $E(L)$ at $L = 200, 300, 400, 600$ with fixed $\chi = 1000$.
Fit $E(L) = E_\infty + a/L^2$ (second-order FD error) to extrapolate $E_\infty$.

### Step 3: Bond Dimension Convergence

For each system at fixed $L$, compute $E(\chi)$ at $\chi = 200, 400, 800, 1200$.
Accept when $|E(\chi) - E(\chi/2)| < 10^{-8}$.

### Step 4: Multiple Random Initializations

For $N \geq 5$, run 3–5 independent DMRG runs with different random initial MPS.
Report the minimum energy; significant spread indicates local minima issues.

---

## Expected Results

| System | Expected DMRG accuracy | Estimated $\chi$ needed | Estimated wall time (single core) |
|--------|------------------------|-------------------------|------------------------------------|
| H      | $10^{-10}$             | 50                      | < 1 min                            |
| He     | $10^{-10}$             | 100                     | < 5 min                            |
| Li     | $10^{-8}$              | 200                     | ~10 min                            |
| Be     | $10^{-8}$              | 300                     | ~30 min                            |
| B      | $10^{-7}$              | 500                     | ~1 hr                              |
| C      | $10^{-7}$              | 800                     | ~3 hr                              |
| N      | $10^{-6}$              | 1000                    | ~6 hr                              |
| O      | $10^{-6}$              | 1200                    | ~12 hr                             |

All estimates assume $L=400$, single CPU core. ITensor.jl parallelizes well with Julia threads.

---

## Key Risks and Mitigations

**Long-range MPO bond dimension**: Naive construction gives $O(L)$ bond dimension. Mitigation: implement SOG-MPO explicitly; $L_\text{SOG} \approx 28$ terms keep MPO bond dimension $O(30)$.

**Local minima for $N \geq 5$**: Strong correlations can trap DMRG. Mitigation: use noise term in early sweeps, multiple random initializations, and subspace expansion.

**Grid discretization error**: Second-order FD introduces $O(\delta^2)$ error. Mitigation: Richardson extrapolation in $L$, or switch to spectral discretization (Chebyshev/Legendre grid) which converges exponentially in $L$ and is more consistent with SG-CI baseline.

**Spin sector**: Verify QN conservation selects correct $N_\uparrow$, $N_\downarrow$. For open-shell systems (B, N), ground state spin may need investigation.

---

## References

- Wu et al., arXiv:2603.23897 (2026) — SOG-TNN paper, source of parameters and extrapolated reference energies
- Stoudenmire, Wagner, White, Burke, *Phys. Rev. Lett.* 109, 056402 (2012) — DMRG for 1D soft-Coulomb systems
- Wagner et al., *Phys. Chem. Chem. Phys.* 14, 8581 (2012) — reference electronic structure in 1D
- ITensor.jl documentation: https://itensor.github.io/ITensors.jl/
