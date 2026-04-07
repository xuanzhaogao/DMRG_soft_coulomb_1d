using ITensors
using ITensorMPS
using HDF5
using Printf

# GPU support: built-in CUDA backend (not deprecated ITensorGPU)
const _WANT_GPU = parse(Bool, get(ENV, "USE_GPU", "false"))
if _WANT_GPU
    using CUDA
end
const USE_GPU = _WANT_GPU && @isdefined(CUDA) && CUDA.functional()
if USE_GPU
    @info "GPU mode: $(CUDA.name(CUDA.device())), $(CUDA.totalmem(CUDA.device()) ÷ 1024^2) MiB"
elseif _WANT_GPU
    @warn "USE_GPU=true but CUDA not functional — falling back to CPU"
end

# QN conservation uses block-sparse tensors. GPU block-sparse support is experimental.
# Set USE_QN=false to use dense tensors on GPU if block-sparse fails.
const USE_QN = parse(Bool, get(ENV, "USE_QN", "true"))

# ============================================================
# Helpers
# ============================================================
soft_coulomb(u) = 1.0 / sqrt(1.0 + u^2)

function spread(n::Int, lo::Int, hi::Int)
    n == 0 && return Int[]
    n == 1 && return [div(lo + hi, 2)]
    round.(Int, range(lo, hi, length=n))
end

function build_hamiltonian(L::Int, delta::Float64, rc::Real, z::Int,
                            r::Vector{Float64}; N::Int=2)
    sites = siteinds("Electron", L; conserve_qns=USE_QN)
    os = OpSum()
    for i in 1:L
        os += 1.0/delta^2, "Nup", i
        os += 1.0/delta^2, "Ndn", i
    end
    for i in 1:L-1
        os += -1.0/(2*delta^2), "Cdagup", i,   "Cup",    i+1
        os += -1.0/(2*delta^2), "Cdagup", i+1, "Cup",    i
        os += -1.0/(2*delta^2), "Cdagdn", i,   "Cdn",    i+1
        os += -1.0/(2*delta^2), "Cdagdn", i+1, "Cdn",    i
    end
    for i in 1:L
        v_en = -z * soft_coulomb(abs(r[i]))
        os += v_en, "Nup", i
        os += v_en, "Ndn", i
    end
    if N > 1
        for i in 1:L
            os += soft_coulomb(0.0), "Nupdn", i
        end
        for i in 1:L, j in i+1:L
            v_ee = soft_coulomb(abs(r[i] - r[j]))
            os += v_ee, "Nup", i, "Nup", j
            os += v_ee, "Ndn", i, "Ndn", j
            os += v_ee, "Nup", i, "Ndn", j
            os += v_ee, "Ndn", i, "Nup", j
        end
    end
    H = MPO(os, sites)
    return H, sites
end

function make_initial_mps(sites, L::Int, Nup::Int, Ndn::Int)
    state = fill("Emp", L)
    up_pos = spread(Nup, 1, L)
    for i in up_pos; state[i] = "Up"; end
    candidates = setdiff(1:L, up_pos)
    dn_pos = isempty(candidates) ? spread(Ndn, 1, L) :
             candidates[spread(Ndn, 1, length(candidates))]
    for i in dn_pos; state[i] = (state[i] == "Up") ? "UpDn" : "Dn"; end
    return MPS(sites, state)
end

function run_dmrg(H, psi0; max_bond::Int=500, cutoff::Float64=1e-12, nsweeps::Int=25)
    ramp = [50, 100, 200, 400, 600, 800, 1000]
    dims  = min.(vcat(ramp, fill(max_bond, max(nsweeps-length(ramp), 0)))[1:nsweeps], max_bond)
    noise = vcat([1e-5, 1e-6, 1e-7, 1e-8], fill(0.0, max(nsweeps-4, 0)))[1:nsweeps]
    sweeps = Sweeps(nsweeps)
    setmaxdim!(sweeps, dims...)
    setcutoff!(sweeps, fill(cutoff, nsweeps)...)
    setnoise!(sweeps, noise...)

    if USE_GPU
        H_run   = cu(H)
        psi_run = cu(psi0)
    else
        H_run   = H
        psi_run = psi0
    end

    energy, psi_out = dmrg(H_run, psi_run, sweeps; outputlevel=1)

    psi_cpu = USE_GPU ? cpu(psi_out) : psi_out
    return energy, psi_cpu
end

# ============================================================
# Study 1: He bond-dimension convergence (fixed δ=0.1)
# ============================================================
function study_bond_dim()
    name, N, z, Nup, Ndn, rc = "He", 2, 2, 1, 1, 13
    delta = 0.1
    L = round(Int, 2*rc / delta)
    r = [-rc + (i-1)*delta for i in 1:L]
    ref = -2.2382578241080

    @printf "\n=== He bond-dim convergence  (δ=%.2f, L=%d) ===\n" delta L
    @printf "%-8s  %-22s  %-8s  %-12s  %-12s\n" "max_χ" "Energy" "act_χ" "ΔE(prev)" "err_vs_ref"
    @printf "%s\n" repeat("-", 70)
    flush(stdout)

    H, sites = build_hamiltonian(L, delta, rc, z, r; N=N)
    @printf "MPO bond dim = %d\n" maxlinkdim(H)
    flush(stdout)

    prev_E = NaN
    results = Tuple{Int,Float64,Int}[]

    for χ in [50, 100, 200, 500, 1000]
        psi0 = make_initial_mps(sites, L, Nup, Ndn)
        E, psi = run_dmrg(H, psi0; max_bond=χ, nsweeps=25)
        act_χ = maxlinkdim(psi)
        dE = isnan(prev_E) ? NaN : E - prev_E
        @printf "%-8d  %22.13f  %-8d  %-12s  %+.3e\n" χ E act_χ \
            (isnan(dE) ? "     —" : @sprintf("%+.3e", dE)) (E - ref)
        flush(stdout)
        push!(results, (χ, E, act_χ))
        prev_E = E
        !isnan(dE) && abs(dE) < 1e-10 && (@printf "  → Converged at χ=%d\n" χ; break)
    end
    return results
end

# ============================================================
# Study 2: He grid-spacing convergence (fixed χ=500)
# ============================================================
function study_grid()
    name, N, z, Nup, Ndn, rc = "He", 2, 2, 1, 1, 13
    ref = -2.2382578241080
    χ = 500

    @printf "\n=== He grid convergence  (χ=%d) ===\n" χ
    @printf "%-8s  %-6s  %-22s  %-12s\n" "δ" "L" "Energy" "err_vs_ref"
    @printf "%s\n" repeat("-", 55)
    flush(stdout)

    results = Tuple{Float64,Int,Float64}[]
    for delta in [0.2, 0.1, 0.05]
        L = round(Int, 2*rc / delta)
        r = [-rc + (i-1)*delta for i in 1:L]
        H, sites = build_hamiltonian(L, delta, rc, z, r; N=N)
        @printf "  δ=%.2f  L=%d  MPO bond dim=%d  " delta L maxlinkdim(H)
        flush(stdout)
        psi0 = make_initial_mps(sites, L, Nup, Ndn)
        E, _ = run_dmrg(H, psi0; max_bond=χ, nsweeps=25)
        @printf "\n%-8.4f  %-6d  %22.13f  %+.3e\n" delta L E (E - ref)
        flush(stdout)
        push!(results, (delta, L, E))
    end
    return results
end

# ============================================================
# Study 3: Li ground state
# ============================================================
function study_Li()
    name, N, z, Nup, Ndn, rc = "Li", 3, 3, 2, 1, 20
    delta = 0.1
    L = round(Int, 2*rc / delta)
    r = [-rc + (i-1)*delta for i in 1:L]
    ref = -4.210531647613249
    χ = parse(Int, get(ENV, "LI_MAXBOND", "1000"))

    @printf "\n=== Li  (δ=%.2f, L=%d, χ=%d) ===\n" delta L χ
    flush(stdout)

    H, sites = build_hamiltonian(L, delta, rc, z, r; N=N)
    @printf "MPO bond dim = %d\n" maxlinkdim(H)
    flush(stdout)

    psi0 = make_initial_mps(sites, L, Nup, Ndn)
    @time E, psi = run_dmrg(H, psi0; max_bond=χ, nsweeps=30, cutoff=1e-12)

    act_χ = maxlinkdim(psi)
    @printf "\nLi  E = %.13f   act_χ = %d\n" E act_χ
    @printf "Li  ref = %.13f   err = %+.3e\n" ref (E - ref)
    flush(stdout)

    h5open("data/Li_psi.h5", "w") do f
        write(f, "psi", psi)
        write(f, "energy", E)
    end
    return E
end

# ============================================================
# Main
# ============================================================
function main()
    mode = get(ENV, "DMRG_MODE", "all")

    (mode == "bondconv" || mode == "all") && study_bond_dim()
    (mode == "gridconv" || mode == "all") && study_grid()
    (mode == "Li"       || mode == "all") && study_Li()
end

main()
