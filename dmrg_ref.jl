using ITensors
using ITensorMPS
using HDF5
using Printf
# Note: ITensorGPU incompatible with ITensors v0.9 / Julia 1.12 — CPU only

# ============================================================
# 1D soft-Coulomb DMRG reference calculations
# Hamiltonian: H = sum_i [-1/2 ∇²_i - z*v(|r_i|)] + sum_{i<j} v(r_{ij})
# v(u) = 1/sqrt(1 + u²)
# ============================================================

function soft_coulomb(u)
    return 1.0 / sqrt(1.0 + u^2)
end

# ============================================================
# Build MPO Hamiltonian via OpSum
# ============================================================
function build_hamiltonian(L::Int, delta::Float64, rc::Real, z::Int, r::Vector{Float64}; N::Int=2)
    sites = siteinds("Electron", L; conserve_qns=true)
    os = OpSum()

    # Kinetic energy: diagonal 1/δ² terms
    for i in 1:L
        os += 1.0/delta^2, "Nup", i
        os += 1.0/delta^2, "Ndn", i
    end
    # Kinetic energy: off-diagonal hopping
    for i in 1:L-1
        os += -1.0/(2*delta^2), "Cdagup", i, "Cup",   i+1
        os += -1.0/(2*delta^2), "Cdagup", i+1, "Cup", i
        os += -1.0/(2*delta^2), "Cdagdn", i, "Cdn",   i+1
        os += -1.0/(2*delta^2), "Cdagdn", i+1, "Cdn", i
    end

    # Electron-nucleus attraction
    for i in 1:L
        v_en = -z * soft_coulomb(abs(r[i]))
        os += v_en, "Nup", i
        os += v_en, "Ndn", i
    end

    # Electron-electron repulsion (exact pairwise; skip for single-electron systems)
    if N > 1
        # On-site Hubbard-U: two electrons of opposite spin at the same site
        # interact with v(0) = 1/sqrt(1+0^2) = 1
        for i in 1:L
            os += soft_coulomb(0.0), "Nupdn", i
        end
        # Inter-site repulsion
        for i in 1:L
            for j in i+1:L
                v_ee = soft_coulomb(abs(r[i] - r[j]))
                os += v_ee, "Nup", i, "Nup", j
                os += v_ee, "Ndn", i, "Ndn", j
                os += v_ee, "Nup", i, "Ndn", j
                os += v_ee, "Ndn", i, "Nup", j
            end
        end
    end

    H = MPO(os, sites)
    return H, sites
end

# ============================================================
# Initial MPS with correct quantum numbers
# ============================================================

# LinRange(1, L, 1) errors in Julia 1.12 when start≠stop, so use this helper
function spread(n::Int, lo::Int, hi::Int)
    n == 0 && return Int[]
    n == 1 && return [div(lo + hi, 2)]
    round.(Int, range(lo, hi, length=n))
end

function make_initial_mps(sites, L::Int, Nup::Int, Ndn::Int)
    state = fill("Emp", L)
    up_pos = spread(Nup, 1, L)
    for i in up_pos
        state[i] = "Up"
    end
    # place down electrons at midpoints between up positions
    candidates = setdiff(1:L, up_pos)
    dn_pos = isempty(candidates) ? spread(Ndn, 1, L) :
             candidates[spread(Ndn, 1, length(candidates))]
    for i in dn_pos
        state[i] = (state[i] == "Up") ? "UpDn" : "Dn"
    end
    return MPS(sites, state)
end

# ============================================================
# DMRG run
# ============================================================
function run_dmrg(H, psi0; max_bond::Int=500, cutoff::Float64=1e-12, nsweeps::Int=20)
    ramp = [50, 100, 200, 400, 600, 800, 1000]
    dims = vcat(ramp, fill(max_bond, max(nsweeps - length(ramp), 0)))[1:nsweeps]
    noise = vcat([1e-5, 1e-6, 1e-7, 1e-8], fill(0.0, max(nsweeps-4, 0)))[1:nsweeps]

    sweeps = Sweeps(nsweeps)
    setmaxdim!(sweeps, dims...)
    setcutoff!(sweeps, fill(cutoff, nsweeps)...)
    setnoise!(sweeps, noise...)

    energy, psi = dmrg(H, psi0, sweeps; outputlevel=1)
    return energy, psi
end

# ============================================================
# Main
# ============================================================
function main()
    # Systems: (name, N, z, Nup, Ndn, rc)
    # Start with H and He for validation; extend to heavier atoms
    systems = [
        ("H",  1, 1, 1, 0, 15),
        ("He", 2, 2, 1, 1, 13),
        ("Li", 3, 3, 2, 1, 20),
        ("Be", 4, 4, 2, 2, 20),
        ("B",  5, 5, 3, 2, 20),
        ("C",  6, 6, 3, 3, 20),
        ("N",  7, 7, 4, 3, 20),
        ("O",  8, 8, 4, 4, 20),
    ]

    # Reference energies from Wu et al. (arXiv:2603.23897)
    refs = Dict(
        "H"  => -0.6697771382138,
        "He" => -2.2382578241080,
        "Li" => -4.210531647613249,
        "Be" => -6.78507795170852,
        "B"  => -9.81971246294425,
        "C"  => -13.33154314459545,
        "N"  => -17.29164709368697,
        "O"  => -21.69957056180725,
    )

    # Grid: δ = 0.1 (L = 2*rc/δ)
    delta = 0.1

    # Which systems to run (set via DMRG_SYSTEMS env var, e.g. "H,He")
    run_list = get(ENV, "DMRG_SYSTEMS", "H,He")
    to_run = Set(strip.(split(run_list, ",")))

    # Bond dimension (override via DMRG_MAXBOND)
    max_bond = parse(Int, get(ENV, "DMRG_MAXBOND", "500"))
    nsweeps  = parse(Int, get(ENV, "DMRG_NSWEEPS", "20"))

    results = Dict{String, Float64}()

    for (name, N, z, Nup, Ndn, rc) in systems
        name ∉ to_run && continue

        L = round(Int, 2*rc / delta)
        r = [-rc + (i-1)*delta for i in 1:L]

        @printf "\n%s\n" repeat("=", 60)
        @printf "System: %s  N=%d  z=%d  Nup=%d  Ndn=%d\n" name N z Nup Ndn
        @printf "Grid:   rc=%.0f  L=%d  δ=%.4f\n" rc L delta
        @printf "%s\n" repeat("=", 60)
        flush(stdout)

        @time begin
            H, sites = build_hamiltonian(L, delta, rc, z, r; N=N)
            @printf "MPO bond dim = %d\n" maxlinkdim(H)
            flush(stdout)

            psi0 = make_initial_mps(sites, L, Nup, Ndn)

            energy, psi = run_dmrg(H, psi0; max_bond=max_bond, cutoff=1e-12, nsweeps=nsweeps)
        end

        results[name] = energy
        ref = get(refs, name, NaN)
        err = energy - ref
        @printf "\n%-4s  E = %.13f   ref = %.13f   err = %+.3e\n" name energy ref err
        flush(stdout)

        # Save MPS to HDF5
        h5open("$(name)_psi.h5", "w") do f
            write(f, "psi", psi)
            write(f, "energy", energy)
        end
    end

    # Summary table
    @printf "\n%s\n" repeat("=", 60)
    @printf "%-6s  %-20s  %-20s  %-12s\n" "System" "E_DMRG" "E_ref" "Error"
    @printf "%s\n" repeat("-", 60)
    for (name, energy) in sort(collect(results))
        ref = get(refs, name, NaN)
        @printf "%-6s  %20.13f  %20.13f  %+12.3e\n" name energy ref (energy-ref)
    end
end

main()
