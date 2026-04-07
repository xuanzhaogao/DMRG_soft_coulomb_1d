include("sog_mpo.jl")
using Printf

function test()
    rc = 3.0; delta = 0.5; z = 2; N = 2; L = 12
    r = [-rc + (i-1)*delta for i in 1:L]

    H_qn, sites_qn = build_hamiltonian_sog(L, delta, rc, z, r; N=N, conserve_qns=true)
    H_nq, sites_nq = build_hamiltonian_sog(L, delta, rc, z, r; N=N, conserve_qns=false)

    states = [
        ("↑5↓8 (3hop)", 5, 8),
        ("↑1↓2 (adj)", 1, 2),
        ("↑6↓7 (adj)", 6, 7),
        ("↑1↓12 (far)", 1, 12),
    ]

    @printf "%-20s  %12s  %12s  %12s\n" "State" "QN" "noQN" "Diff"
    for (label, i, j) in states
        st = fill("Emp", L); st[i] = "Up"; st[j] = "Dn"
        E_qn = inner(MPS(sites_qn, st)', H_qn, MPS(sites_qn, st))
        E_nq = inner(MPS(sites_nq, st)', H_nq, MPS(sites_nq, st))
        @printf "%-20s  %12.6f  %12.6f  %+12.2e\n" label E_qn E_nq (E_qn - E_nq)
    end
end

test()
