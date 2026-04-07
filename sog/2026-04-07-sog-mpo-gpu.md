# SOG-MPO Construction + GPU Production Runs

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace O(L²) OpSum-based MPO construction with O(L·K) direct MPO construction using sum-of-exponentials (SOG) decomposition, then run all systems H–O on GPU.

**Architecture:** The soft-Coulomb kernel `1/√(1+t²)` is decomposed into 69 exponential terms via pyvpmr. The e-e interaction MPO is built directly from W-matrices (bypassing OpSum), where each SOG term adds bond states for exponential propagation. Complex conjugate pairs use 2×2 real rotation blocks. One-body terms (kinetic + nuclear) stay in OpSum (only O(L) terms). The two MPOs are summed. GPU runs use CUDA on A100 nodes via slurm.

**Tech Stack:** Julia 1.12.5, ITensors v0.9.25, ITensorMPS v0.3.45, CUDA v5.11, pyvpmr (Python)

---

### Task 1: Generate and Save SOG Coefficients

**Files:**
- Modify: `sog_coeffs.jl` (already generated, restructure for real/conjugate-pair grouping)

SOG coefficients are already generated (69 terms, tol=1e-8, max error 5.53e-4 on [0,40]).
The file contains raw complex arrays `SOG_M_RE`, `SOG_M_IM`, `SOG_S_RE`, `SOG_S_IM`.

- [ ] **Step 1: Rewrite `sog_coeffs.jl` to separate real terms and conjugate pairs**

Restructure the data into:
- `SOG_REAL`: vector of `(m, s)` for purely real terms
- `SOG_PAIR`: vector of `(p, q, α, β)` for conjugate pairs where `m = p+iq`, `s = α+iβ`

This makes the MPO construction cleaner.

- [ ] **Step 2: Verify the restructured coefficients reproduce the kernel**

Write a quick Julia test: evaluate `Σ terms` at t=0.1, 1.0, 10.0, compare to `1/√(1+t²)`.

### Task 2: Build SOG-MPO for e-e Interaction

**Files:**
- Create: `sog_mpo.jl` — contains `build_hamiltonian_sog()` function

The MPO W-matrix at each site l has bond dimension `D = 2 + 4 + N_real + 2*N_pairs`:
- State 0: left identity
- States 1–4: nearest-neighbor hopping (c†↑, c↑, c†↓, c↓)
- States 5 to 5+N_real-1: real SOG propagators
- States 5+N_real to 5+N_real+2*N_pairs-1: conjugate pair propagators (2 states each)
- State D-1: right identity

W-matrix structure (lower-triangular):
```
W[l][0,0] = I                                    (left identity)
W[l][0,1] = Cdagup_l                             (start up-hopping)
W[l][0,2] = Cup_l                                (start reverse up-hopping)
W[l][0,3] = Cdagdn_l                             (start down-hopping)
W[l][0,4] = Cdn_l                                (start reverse down-hopping)
W[l][0, 5+k] = r_k cos(θ_k) Ntot_l              (start real SOG or pair-a)
W[l][0, 5+k+1] = r_k sin(θ_k) Ntot_l            (start pair-b, pairs only)
W[l][k,k] = r_k cos(θ_k) I                       (propagate SOG, diagonal)
W[l][2k-1,2k] = -r_k sin(θ_k) I                  (propagate SOG, rotation)
W[l][2k,2k-1] = r_k sin(θ_k) I                   (propagate SOG, rotation)
W[l][k, D-1] = weight × Ntot_l                   (end SOG chain)
W[l][1, D-1] = -1/(2δ²) Cup_l                    (end up-hop)
W[l][2, D-1] = -1/(2δ²) Cdagup_l                 (end reverse up-hop)
W[l][3, D-1] = -1/(2δ²) Cdn_l                    (end down-hop)
W[l][4, D-1] = -1/(2δ²) Cdagdn_l                 (end reverse down-hop)
W[l][0, D-1] = h_l                               (diagonal: KE + nuclear + on-site)
W[l][D-1, D-1] = I                               (right identity)
```

Where `h_l = (1/δ²)(Nup+Ndn) + v_en(l)(Nup+Ndn) + v(0)*Nupdn`.

- [ ] **Step 1: Write `build_hamiltonian_sog()` that constructs MPO tensors directly**

Implementation: loop over sites l=1..L, construct each ITensor `W[l]` with bond indices `(link_l-1, link_l)` and physical indices `(site_l', site_l)`. Use `op()` to get operator matrices for Nup, Ndn, Ntot, Nupdn, Cdagup, Cup, etc.

- [ ] **Step 2: Test on tiny system (L=12) — check MPO bond dimension matches expectation**

Expected bond dim = 2 + 4 + N_real + 2*N_pairs = 6 + 3 + 2*33 = 75.

### Task 3: Verify SOG-MPO Accuracy Against Direct MPO

**Files:**
- Create: `test_sog_mpo.jl` — comparison test

- [ ] **Step 1: Write test script that builds both naive and SOG MPOs for He (rc=3, δ=0.5, L=12)**

- [ ] **Step 2: Run DMRG with both MPOs (8 sweeps, χ=50) and compare energies**

Success criterion: |E_sog - E_naive| < 1e-3 (SOG error is ~5e-4 in kernel, energy error should be similar).

- [ ] **Step 3: Profile SOG-MPO construction for He (rc=13, δ=0.1, L=260)**

Compare timing: naive ~520s vs SOG ~seconds. Verify SOG-MPO bond dim ≈ 75.

### Task 4: Write Production Script with GPU Support

**Files:**
- Create: `dmrg_sog.jl` — production DMRG script using SOG-MPO + GPU

This combines:
- SOG-MPO construction from Task 2
- GPU support via `cu()` / `cpu()` (tested working in test_gpu.jl)
- All 8 systems H–O with correct parameters
- Environment variables: DMRG_SYSTEMS, DMRG_MAXBOND, DMRG_NSWEEPS, USE_GPU, USE_QN
- HDF5 output for MPS wavefunctions

- [ ] **Step 1: Write `dmrg_sog.jl` combining SOG-MPO + GPU + all systems**

- [ ] **Step 2: Test locally on CPU: H system (quick, <1 min)**

Verify energy matches reference within grid discretization error.

### Task 5: GPU Production Runs on Cluster

**Files:**
- Create: `job_sog_gpu.slurm` — slurm script for A100 GPU runs

- [ ] **Step 1: Write slurm script targeting A100-80GB nodes**

```bash
#SBATCH -p gpu
#SBATCH -C a100-80gb
#SBATCH -N1 -n1 -c 16 --gpus-per-task=1
#SBATCH --time=24:00:00
```

- [ ] **Step 2: Submit H and He first as validation**

Run: `sbatch job_sog_gpu.slurm` with `DMRG_SYSTEMS=H,He`
Check: energies match previous CPU results within SOG error (~5e-4).

- [ ] **Step 3: Submit Li–O sequentially (or as separate jobs)**

Heavier atoms need more sweeps and bond dimension. Expected wall times:
- Li, Be: ~30 min each
- B, C: ~2–4 hrs each  
- N, O: ~8–12 hrs each

- [ ] **Step 4: Collect and summarize results**

Compare all energies against Wu et al. reference values.
