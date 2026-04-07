include("sog_mpo.jl")
using Printf

function test_hopping()
    rc = 0.5; delta = 0.5; z = 2; L = 2
    r = [-rc + (i-1)*delta for i in 1:L]

    # Naive
    sites_n = siteinds("Electron", L; conserve_qns=false)
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
    H_n = MPO(os, sites_n)

    # SOG
    H_s, sites_s = build_hamiltonian_sog(L, delta, rc, z, r; N=2, conserve_qns=false)

    @printf "Testing hopping matrix elements ⟨ψ₁|H|ψ₂⟩:\n\n"

    # Test: ⟨0↑|H|↑0⟩ (up-spin hop from site 1 to site 2)
    test_cases = [
        ("⟨0↑|H|↑0⟩ (up hop)", ["Emp","Up"], ["Up","Emp"]),
        ("⟨↑0|H|↑0⟩ (diagonal)", ["Up","Emp"], ["Up","Emp"]),
        ("⟨0↓|H|↓0⟩ (dn hop)", ["Emp","Dn"], ["Dn","Emp"]),
        ("⟨↓↑|H|↑↓⟩ (swap)", ["Dn","Up"], ["Up","Dn"]),
        ("⟨↑↓|H|↑↓⟩ (diagonal)", ["Up","Dn"], ["Up","Dn"]),
    ]

    for (label, st_bra, st_ket) in test_cases
        bra_n = MPS(sites_n, st_bra)
        ket_n = MPS(sites_n, st_ket)
        bra_s = MPS(sites_s, st_bra)
        ket_s = MPS(sites_s, st_ket)
        En = inner(bra_n', H_n, ket_n)
        Es = inner(bra_s', H_s, ket_s)
        @printf "  %-25s: naive=%+10.6f  sog=%+10.6f  diff=%+.2e\n" label En Es (Es-En)
    end
end

test_hopping()
