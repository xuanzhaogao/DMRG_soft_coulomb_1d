using ITensors, ITensorMPS, Printf

include(joinpath(@__DIR__, "..", "sog", "soe_mpo.jl"))

function main()
    rc = 1.0; delta = 0.5; z = 2; L = 4
    r = [-rc + (i-1)*delta for i in 1:L]

    sites = siteinds("Electron", L; conserve_qns=true)

    # Naive H using SOE kernel via OpSum
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
    for i in 1:L
        os += soft_coulomb(0.0), "Nupdn", i
    end
    for i in 1:L, j in i+1:L
        v_ee = soft_coulomb_soe(abs(r[i] - r[j]))
        os += v_ee, "Nup", i, "Nup", j
        os += v_ee, "Ndn", i, "Ndn", j
        os += v_ee, "Nup", i, "Ndn", j
        os += v_ee, "Ndn", i, "Nup", j
    end
    H_naive = MPO(os, sites)

    # SOE W-matrix H
    H_soe, _ = build_hamiltonian_soe(L, delta, rc, z, r; N=2, sites=sites)

    @printf "Naive bond dim = %d, SOE bond dim = %d\n\n" maxlinkdim(H_naive) maxlinkdim(H_soe)

    # Compare expectation values
    states_list = [
        ["Up", "Dn", "Emp", "Emp"],
        ["Emp", "UpDn", "Emp", "Emp"],
        ["Up", "Emp", "Dn", "Emp"],
        ["Up", "Emp", "Emp", "Dn"],
        ["Emp", "Up", "Dn", "Emp"],
        ["Emp", "Up", "Emp", "Dn"],
        ["Emp", "Emp", "Up", "Dn"],
    ]

    @printf "%-30s %15s %15s %12s\n" "State" "Naive" "SOE" "Diff"
    max_diff = 0.0
    for st in states_list
        psi = MPS(sites, st)
        e_naive = real(inner(psi', H_naive, psi))
        e_soe = real(inner(psi', H_soe, psi))
        diff = e_soe - e_naive
        max_diff = max(max_diff, abs(diff))
        @printf "%-30s %15.8f %15.8f %12.2e\n" join(st, ",") e_naive e_soe diff
    end
    @printf "\nMax |diff| = %.3e\n" max_diff
end

main()
