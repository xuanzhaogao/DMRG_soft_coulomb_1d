include("sog_mpo.jl")
using Printf

function spread(n, lo, hi)
    n == 0 && return Int[]
    n == 1 && return [div(lo + hi, 2)]
    round.(Int, range(lo, hi, length=n))
end
function make_initial_mps(sites, L, Nup, Ndn)
    state = fill("Emp", L)
    for i in spread(Nup, 1, L); state[i] = "Up"; end
    cands = setdiff(1:L, spread(Nup, 1, L))
    for i in (isempty(cands) ? spread(Ndn,1,L) : cands[spread(Ndn,1,length(cands))])
        state[i] = state[i] == "Up" ? "UpDn" : "Dn"
    end
    MPS(sites, state)
end

function cross_eval()
    rc = 3.0; delta = 0.5; z = 2; N = 2; Nup = 1; Ndn = 1
    L = round(Int, 2*rc / delta)
    r = [-rc + (i-1)*delta for i in 1:L]

    sites = siteinds("Electron", L; conserve_qns=true)

    # Naive DMRG to get ground state
    os = OpSum()
    for i in 1:L; os += 1.0/delta^2, "Nup", i; os += 1.0/delta^2, "Ndn", i; end
    for i in 1:L-1
        os += -1.0/(2*delta^2), "Cdagup", i, "Cup", i+1
        os += -1.0/(2*delta^2), "Cdagup", i+1, "Cup", i
        os += -1.0/(2*delta^2), "Cdagdn", i, "Cdn", i+1
        os += -1.0/(2*delta^2), "Cdagdn", i+1, "Cdn", i
    end
    for i in 1:L
        v_en = -z * soft_coulomb(abs(r[i]))
        os += v_en, "Nup", i; os += v_en, "Ndn", i
    end
    for i in 1:L; os += soft_coulomb(0.0), "Nupdn", i; end
    for i in 1:L, j in i+1:L
        v_ee = soft_coulomb(abs(r[i] - r[j]))
        os += v_ee, "Nup", i, "Nup", j; os += v_ee, "Ndn", i, "Ndn", j
        os += v_ee, "Nup", i, "Ndn", j; os += v_ee, "Ndn", i, "Nup", j
    end
    H_naive = MPO(os, sites)

    sweeps_n = Sweeps(15)
    setmaxdim!(sweeps_n, fill(200, 15)...)
    setcutoff!(sweeps_n, fill(1e-12, 15)...)
    setnoise!(sweeps_n, vcat([1e-5, 1e-6, 1e-7, 1e-8], fill(0.0, 11))...)
    psi0 = make_initial_mps(sites, L, Nup, Ndn)
    E_naive, psi_gs = dmrg(H_naive, psi0, sweeps_n; outputlevel=0)
    @printf "Naive GS: E = %.10f, maxlinkdim = %d\n\n" E_naive maxlinkdim(psi_gs)

    # Build SOG MPO with SAME sites
    H_sog, _ = build_hamiltonian_sog(L, delta, rc, z, r; N=N, sites=sites)
    @printf "SOG MPO bond dim = %d\n" maxlinkdim(H_sog)

    # Evaluate naive GS with SOG MPO
    E_sog_at_gs = inner(psi_gs', H_sog, psi_gs)
    @printf "⟨psi_naive_gs|H_sog|psi_naive_gs⟩ = %.10f\n" E_sog_at_gs
    @printf "⟨psi_naive_gs|H_naive|psi_naive_gs⟩ = %.10f\n" E_naive
    @printf "Diff = %+.2e (should be ~5e-4 from SOG kernel error)\n\n" (E_sog_at_gs - E_naive)

    # Now do SOG DMRG starting from the naive GS
    @printf "SOG DMRG from naive GS...\n"
    sweeps_s = Sweeps(10)
    setmaxdim!(sweeps_s, fill(200, 10)...)
    setcutoff!(sweeps_s, fill(1e-12, 10)...)
    setnoise!(sweeps_s, vcat([1e-5, 1e-6, 1e-7], fill(0.0, 7))...)
    E_sog, psi_sog = dmrg(H_sog, psi_gs, sweeps_s; outputlevel=1)
    @printf "\nSOG DMRG result: E = %.10f\n" E_sog
    @printf "Diff from naive GS: %+.2e\n" (E_sog - E_naive)
end

cross_eval()
