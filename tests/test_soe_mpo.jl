using ITensors
using ITensorMPS
using Printf

# Load SOE-MPO builder (path relative to project root)
include(joinpath(@__DIR__, "..", "sog", "soe_mpo.jl"))

function build_hamiltonian_naive(L, delta, rc, z, r; N=2)
    sites = siteinds("Electron", L; conserve_qns=true)
    os = OpSum()
    for i in 1:L
        os += 1.0/delta^2, "Nup", i
        os += 1.0/delta^2, "Ndn", i
    end
    for i in 1:L-1
        os += -1.0/(2*delta^2), "Cdagup", i, "Cup", i+1
        os += -1.0/(2*delta^2), "Cdagup", i+1, "Cup", i
        os += -1.0/(2*delta^2), "Cdagdn", i, "Cdn", i+1
        os += -1.0/(2*delta^2), "Cdagdn", i+1, "Cdn", i
    end
    for i in 1:L
        v_en = -z * soft_coulomb(abs(r[i]))
        os += v_en, "Nup", i
        os += v_en, "Ndn", i
    end
    if N > 1
        for i in 1:L; os += soft_coulomb(0.0), "Nupdn", i; end
        for i in 1:L, j in i+1:L
            v_ee = soft_coulomb(abs(r[i] - r[j]))
            os += v_ee, "Nup", i, "Nup", j
            os += v_ee, "Ndn", i, "Ndn", j
            os += v_ee, "Nup", i, "Ndn", j
            os += v_ee, "Ndn", i, "Nup", j
        end
    end
    return MPO(os, sites), sites
end

function spread(n::Int, lo::Int, hi::Int)
    n == 0 && return Int[]
    n == 1 && return [div(lo + hi, 2)]
    round.(Int, range(lo, hi, length=n))
end

function make_initial_mps(sites, L, Nup, Ndn)
    state = fill("Emp", L)
    up_pos = spread(Nup, 1, L)
    for i in up_pos; state[i] = "Up"; end
    candidates = setdiff(1:L, up_pos)
    dn_pos = isempty(candidates) ? spread(Ndn, 1, L) :
             candidates[spread(Ndn, 1, length(candidates))]
    for i in dn_pos; state[i] = (state[i] == "Up") ? "UpDn" : "Dn"; end
    return MPS(sites, state)
end

function run_dmrg_quick(H, psi0; max_bond=50, nsweeps=10)
    dims  = fill(max_bond, nsweeps)
    noise = vcat([1e-5, 1e-6, 1e-7, 1e-8], fill(0.0, nsweeps - 4))
    sweeps = Sweeps(nsweeps)
    setmaxdim!(sweeps, dims...)
    setcutoff!(sweeps, fill(1e-10, nsweeps)...)
    setnoise!(sweeps, noise...)
    return dmrg(H, psi0, sweeps; outputlevel=1)
end

function test_small()
    rc = 3.0; delta = 0.5; z = 2; N = 2; Nup = 1; Ndn = 1
    L = round(Int, 2*rc / delta)
    r = [-rc + (i-1)*delta for i in 1:L]

    @printf "\n=== He (rc=%.1f, δ=%.1f, L=%d) ===\n" rc delta L

    # Naive (QN-conserving, gold standard)
    @printf "[Naive+QN] Building MPO... "
    flush(stdout)
    t1 = @elapsed (H_n, sites_n) = build_hamiltonian_naive(L, delta, rc, z, r; N=N)
    @printf "%.2f s (bond dim=%d)\n" t1 maxlinkdim(H_n)
    flush(stdout)
    psi0_n = make_initial_mps(sites_n, L, Nup, Ndn)
    E_n, _ = run_dmrg_quick(H_n, psi0_n)
    @printf "[Naive+QN] E = %.10f\n\n" E_n
    flush(stdout)

    # SOE (no QN, dense tensors)
    @printf "[SOE] Building MPO... "
    flush(stdout)
    t2 = @elapsed (H_s, sites_s) = build_hamiltonian_soe(L, delta, rc, z, r; N=N)
    @printf "%.2f s (bond dim=%d)\n" t2 maxlinkdim(H_s)
    flush(stdout)
    psi0_s = make_initial_mps(sites_s, L, Nup, Ndn)
    E_s, psi_s = run_dmrg_quick(H_s, psi0_s)

    ntot = sum(expect(psi_s, "Ntot"))
    sz   = sum(expect(psi_s, "Sz"))
    @printf "[SOE] E = %.10f, ⟨Ntot⟩=%.4f, ⟨Sz⟩=%.4f\n" E_s ntot sz
    flush(stdout)

    diff = abs(E_s - E_n)
    @printf "\nΔE = %.2e  (SOE kernel max error ~7.5e-7)\n" diff
    @printf "Status: %s\n" (diff < 1e-3 ? "PASS" : "FAIL")
end

function test_profile()
    rc = 13.0; delta = 0.1; z = 2; N = 2; Nup = 1; Ndn = 1
    L = round(Int, 2*rc / delta)
    r = [-rc + (i-1)*delta for i in 1:L]

    @printf "\n=== He profile (rc=%.1f, δ=%.1f, L=%d) ===\n" rc delta L
    flush(stdout)

    @printf "[SOE] Building MPO... "
    flush(stdout)
    t = @elapsed (H, sites) = build_hamiltonian_soe(L, delta, rc, z, r; N=N)
    @printf "%.2f s (bond dim=%d)\n" t maxlinkdim(H)
    flush(stdout)

    psi0 = make_initial_mps(sites, L, Nup, Ndn)
    sweeps = Sweeps(5)
    setmaxdim!(sweeps, 50, 100, 200, 200, 200)
    setcutoff!(sweeps, fill(1e-10, 5)...)
    setnoise!(sweeps, 1e-5, 1e-6, 1e-7, 0.0, 0.0)
    @printf "[SOE] DMRG (5 sweeps, χ=200)...\n"
    flush(stdout)
    t_dmrg = @elapsed begin
        E, psi = dmrg(H, psi0, sweeps; outputlevel=1)
    end

    ntot = sum(expect(psi, "Ntot"))
    sz   = sum(expect(psi, "Sz"))
    @printf "\n[SOE] E = %.10f, ⟨Ntot⟩=%.4f, ⟨Sz⟩=%.4f\n" E ntot sz
    @printf "Ref:   E = -2.2382578241\n"
    @printf "DMRG time: %.1f s\n" t_dmrg
    flush(stdout)
end

test_small()
test_profile()
