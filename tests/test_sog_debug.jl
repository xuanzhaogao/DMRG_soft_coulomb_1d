using ITensors
using ITensorMPS
using Printf

include("sog_mpo.jl")

function debug_test()
    rc = 1.5; delta = 0.5; z = 2; N = 2
    L = round(Int, 2*rc / delta)  # L=6
    r = [-rc + (i-1)*delta for i in 1:L]
    @printf "L=%d, r=%s\n" L r

    # Naive MPO (QN)
    sites_n = siteinds("Electron", L; conserve_qns=true)
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
    for i in 1:L; os += soft_coulomb(0.0), "Nupdn", i; end
    for i in 1:L, j in i+1:L
        v_ee = soft_coulomb(abs(r[i] - r[j]))
        os += v_ee, "Nup", i, "Nup", j
        os += v_ee, "Ndn", i, "Ndn", j
        os += v_ee, "Nup", i, "Ndn", j
        os += v_ee, "Ndn", i, "Nup", j
    end
    H_naive = MPO(os, sites_n)

    # SOG (no QN for direct comparison)
    H_sog, sites_s = build_hamiltonian_sog(L, delta, rc, z, r; N=N, conserve_qns=false)

    @printf "%-20s  %12s  %12s  %12s\n" "State" "Naive" "SOG" "Diff"
    @printf "%s\n" repeat("-", 62)

    states = [
        ("↑1↓2 (adj,bndry)",  ["Up","Dn","Emp","Emp","Emp","Emp"]),
        ("↑2↓3 (adj,inner)",  ["Emp","Up","Dn","Emp","Emp","Emp"]),
        ("↑3↓4 (adj,inner)",  ["Emp","Emp","Up","Dn","Emp","Emp"]),
        ("↑5↓6 (adj,bndry)",  ["Emp","Emp","Emp","Emp","Up","Dn"]),
        ("↑1↓3 (2hop,bndry)", ["Up","Emp","Dn","Emp","Emp","Emp"]),
        ("↑2↓4 (2hop,inner)", ["Emp","Up","Emp","Dn","Emp","Emp"]),
        ("↑1↓4 (3hop,bndry)", ["Up","Emp","Emp","Dn","Emp","Emp"]),
        ("↑2↓5 (3hop,inner)", ["Emp","Up","Emp","Emp","Dn","Emp"]),
        ("↑1↓6 (5hop,bndry)", ["Up","Emp","Emp","Emp","Emp","Dn"]),
    ]

    for (label, st) in states
        psi_n = MPS(sites_n, st)
        psi_s = MPS(sites_s, st)
        En = inner(psi_n', H_naive, psi_n)
        Es = inner(psi_s', H_sog, psi_s)
        @printf "%-20s  %12.6f  %12.6f  %+12.2e\n" label En Es (Es - En)
    end
end

debug_test()
