include("sog_mpo.jl")
using Printf

function check()
    rc = 0.75; delta = 0.5; z = 2; N = 2; L = 3
    r = [-rc + (i-1)*delta for i in 1:L]
    H, sites = build_hamiltonian_sog(L, delta, rc, z, r; N=N, conserve_qns=false)

    D = 75
    link1 = inds(H[2])[3]
    link2 = inds(H[2])[4]
    s2 = sites[2]

    # Check M2[SOG_k, 71] at empty site — should ALL be 0
    @printf "Checking M2[SOG_k, POS_RID=71] at empty site 2:\n"
    max_leak = 0.0
    for a in 2:70
        v = H[2][s2' => 1, dag(s2) => 1, link1 => a, link2 => 71]
        if abs(v) > 1e-15
            @printf "  LEAK! M2[%d, 71] = %.10f\n" a v
            max_leak = max(max_leak, abs(v))
        end
    end
    @printf "Max leak into RID: %.2e\n\n" max_leak

    # Check M2[1, SOG_k] at empty site — should ALL be 0
    @printf "Checking M2[POS_LID=1, SOG_k] at empty site 2:\n"
    max_leak2 = 0.0
    for b in 2:70
        v = H[2][s2' => 1, dag(s2) => 1, link1 => 1, link2 => b]
        if abs(v) > 1e-15
            @printf "  LEAK! M2[1, %d] = %.10f\n" b v
            max_leak2 = max(max_leak2, abs(v))
        end
    end
    @printf "Max leak from LID: %.2e\n\n" max_leak2

    # Now check M2 at a state with 1 electron: |↑⟩ = state 2
    @printf "Checking M2 at occupied site (|↑⟩, d=2):\n"
    @printf "  M2[1, 71] (diagonal energy): %.6f\n" H[2][s2'=>2, dag(s2)=>2, link1=>1, link2=>71]
    @printf "  M2[1, 2] (SOG start, Ntot=1): %.6f\n" H[2][s2'=>2, dag(s2)=>2, link1=>1, link2=>2]

    # Check: does M2[SOG_k, 71] leak at occupied site?
    total_end = 0.0
    for a in 2:70
        v = H[2][s2' => 2, dag(s2) => 2, link1 => a, link2 => 71]
        total_end += v
    end
    @printf "  Σ M2[SOG_k, 71] for |↑⟩: %.6f  (SOG ends at occupied site)\n" total_end

    # This is the KEY question: at an OCCUPIED intermediate site, the SOG channels
    # have BOTH propagation (SOG_k → SOG_k) AND ending (SOG_k → RID).
    # The ending leaks the accumulated SOG value into the RID channel!
    @printf "\n=== DIAGNOSIS ===\n"
    @printf "At occupied intermediate sites, SOG end transition (SOG_k → RID)\n"
    @printf "adds wt_k × Ntot to the RID channel. This is CORRECT for the\n"
    @printf "e-e pair ending at this site. But at EMPTY intermediate sites,\n"
    @printf "this is 0 (Ntot=0). So far so good.\n\n"

    # But what about the START at intermediate sites?
    @printf "At intermediate sites, SOG start (LID → SOG_k) adds λ_k × Ntot.\n"
    @printf "At EMPTY sites: λ_k × 0 = 0. At OCCUPIED sites: λ_k × Ntot.\n"
    @printf "This starts NEW SOG chains at each occupied site.\n\n"

    # The issue might be: for state |↑, 0, ↓⟩, there are no occupied intermediate sites.
    # So no spurious starts or ends. The error must come from somewhere else.
    #
    # Let me verify: compute sog_part directly
    sog_part = 0.0
    for k in 2:70
        v1k = H[1][sites[1]' => 2, dag(sites[1]) => 2, link1 => k]
        v2k = sum(H[2][s2' => 1, dag(s2) => 1, link1 => a, link2 => k] *
                  H[1][sites[1]' => 2, dag(sites[1]) => 2, link1 => a]
                  for a in 1:D)
        v3k = H[3][sites[3]' => 3, dag(sites[3]) => 3, link2 => k]
        sog_part += v2k * v3k
    end
    @printf "sog_part (contracted) = %.10f\n" sog_part

    # Also compute just the SOG channel contribution
    sog_isolated = 0.0
    for k in 2:70
        v1k = H[1][sites[1]' => 2, dag(sites[1]) => 2, link1 => k]
        v2kk = H[2][s2' => 1, dag(s2) => 1, link1 => k, link2 => k]
        v3k = H[3][sites[3]' => 3, dag(sites[3]) => 3, link2 => k]
        sog_isolated += v1k * v2kk * v3k
    end
    @printf "sog_isolated (diagonal only) = %.10f\n" sog_isolated

    # Check pair off-diagonal contributions
    sog_offdiag = 0.0
    for k in 1:N_SOG_PAIR
        ia = 1 + N_SOG_REAL + 2*(k-1) + 1
        ib = ia + 1
        v1a = H[1][sites[1]'=>2, dag(sites[1])=>2, link1=>ia]
        v1b = H[1][sites[1]'=>2, dag(sites[1])=>2, link1=>ib]
        # Propagation
        Maa = H[2][s2'=>1, dag(s2)=>1, link1=>ia, link2=>ia]
        Mab = H[2][s2'=>1, dag(s2)=>1, link1=>ia, link2=>ib]
        Mba = H[2][s2'=>1, dag(s2)=>1, link1=>ib, link2=>ia]
        Mbb = H[2][s2'=>1, dag(s2)=>1, link1=>ib, link2=>ib]
        v2a = Maa*v1a + Mba*v1b
        v2b = Mab*v1a + Mbb*v1b
        v3a = H[3][sites[3]'=>3, dag(sites[3])=>3, link2=>ia]
        v3b = H[3][sites[3]'=>3, dag(sites[3])=>3, link2=>ib]
        sog_offdiag += v2a*v3a + v2b*v3b
    end
    # Add real terms
    for k in 1:N_SOG_REAL
        idx = 1 + k
        v1k = H[1][sites[1]'=>2, dag(sites[1])=>2, link1=>idx]
        v2kk = H[2][s2'=>1, dag(s2)=>1, link1=>idx, link2=>idx]
        v3k = H[3][sites[3]'=>3, dag(sites[3])=>3, link2=>idx]
        sog_offdiag += v1k * v2kk * v3k
    end
    @printf "sog_full (with off-diagonal) = %.10f\n" sog_offdiag
    @printf "Expected v_sog(1.0) ≈ 0.7076\n"
end

check()
