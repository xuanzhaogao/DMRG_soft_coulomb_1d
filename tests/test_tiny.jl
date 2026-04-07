using ITensors
using ITensorMPS
using Printf

include("sog_mpo.jl")

function trace_contraction()
    rc = 0.75; delta = 0.5; z = 2; N = 2; L = 3
    r = [-rc + (i-1)*delta for i in 1:L]
    @printf "L=%d, r=%s\n\n" L r

    H, sites = build_hamiltonian_sog(L, delta, rc, z, r; N=N, conserve_qns=false)

    D = 75
    link1 = inds(H[2])[3]  # links[1], shared between H[1] and H[2]
    link2 = inds(H[2])[4]  # links[2], shared between H[2] and H[3]

    # Site 1: ⟨↑|W[1]|↑⟩ → vector indexed by link1
    s1 = sites[1]
    v1 = zeros(D)
    for b in 1:D
        v1[b] = H[1][s1' => 2, dag(s1) => 2, link1 => b]
    end
    @printf "Site 1 (|↑⟩): v1\n"
    @printf "  v1[1] (LID) = %.8f\n" v1[1]
    @printf "  v1[71] (RID) = %.8f  ← diagonal energy h₁\n" v1[71]
    sog_sum = sum(v1[2:70])
    @printf "  Σ v1[2:70] (SOG starts) = %.8f\n" sog_sum
    @printf "  v1[72:75] (hop starts) = %s\n" v1[72:75]

    # Site 2: ⟨0|W[2]|0⟩ → matrix indexed by (link1, link2)
    s2 = sites[2]
    M2 = zeros(D, D)
    for a in 1:D, b in 1:D
        M2[a,b] = H[2][s2' => 1, dag(s2) => 1, link1 => a, link2 => b]
    end
    @printf "\nSite 2 (|0⟩): M2 — key elements:\n"
    @printf "  M2[1,1] (LID→LID) = %.8f\n" M2[1,1]
    @printf "  M2[71,71] (RID→RID) = %.8f\n" M2[71,71]
    @printf "  M2[1,71] (LID→RID, should be 0 at empty) = %.8f\n" M2[1,71]
    @printf "  M2[2,2] (SOG real 1 propagate) = %.8f\n" M2[2,2]
    @printf "  M2[3,3] (SOG real 2 propagate) = %.8f\n" M2[3,3]
    # Check for unexpected non-zero elements
    unexpected = 0
    for a in 1:D, b in 1:D
        if abs(M2[a,b]) > 1e-15
            # Expected: diagonal (a==b) for propagation, (1,b) for starts, (a,71) for ends
            is_diag = (a == b)
            is_start = (a == 1)
            is_end = (b == 71)
            is_pair_offdiag = false
            for k in 1:N_SOG_PAIR
                ia = 1 + N_SOG_REAL + 2*(k-1) + 1
                ib = ia + 1
                if (a == ia && b == ib) || (a == ib && b == ia)
                    is_pair_offdiag = true
                end
            end
            if !is_diag && !is_start && !is_end && !is_pair_offdiag
                unexpected += 1
                if unexpected <= 5
                    @printf "  UNEXPECTED M2[%d,%d] = %.8f\n" a b M2[a,b]
                end
            end
        end
    end
    @printf "  Unexpected non-zero elements: %d\n" unexpected

    # v2 = v1 * M2
    v2 = M2' * v1  # v2[b] = Σ_a v1[a] * M2[a,b]
    @printf "\nAfter propagation through site 2: v2 = v1 × M2\n"
    @printf "  v2[1] (LID) = %.8f  ← should be v1[1]=1\n" v2[1]
    @printf "  v2[71] (RID) = %.8f  ← should be v1[71]=h₁\n" v2[71]
    sog_sum2 = sum(v2[2:70])
    @printf "  Σ v2[2:70] (SOG after propagation) = %.8f\n" sog_sum2

    # Site 3: ⟨↓|W[3]|↓⟩ → vector indexed by link2
    s3 = sites[3]
    v3 = zeros(D)
    for a in 1:D
        v3[a] = H[3][s3' => 3, dag(s3) => 3, link2 => a]
    end
    @printf "\nSite 3 (|↓⟩): v3\n"
    @printf "  v3[1] (LID) = %.8f  ← diagonal energy h₃\n" v3[1]
    @printf "  v3[71] (RID) = %.8f  ← identity\n" v3[71]
    sog_end = sum(v3[2:70])
    @printf "  Σ v3[2:70] (SOG ends) = %.8f\n" sog_end

    # Final energy
    E = dot(v2, v3)
    @printf "\nEnergy = v2 · v3 = %.8f\n" E
    @printf "  = v2[1]*v3[1] + v2[71]*v3[71] + Σ(SOG)\n"
    @printf "  = %.8f * %.8f + %.8f * %.8f + sog_part\n" v2[1] v3[1] v2[71] v3[71]
    @printf "  = %.8f + %.8f + sog_part\n" v2[1]*v3[1] v2[71]*v3[71]
    sog_part = sum(v2[k]*v3[k] for k in 2:70)
    @printf "  sog_part = %.8f\n" sog_part
    @printf "  Total = %.8f\n" (v2[1]*v3[1] + v2[71]*v3[71] + sog_part)

    # Sanity: what is the expected one-body energy?
    h1 = 1.0/delta^2 + (-z * soft_coulomb(abs(r[1])))
    h3 = 1.0/delta^2 + (-z * soft_coulomb(abs(r[3])))
    @printf "\nExpected: h₁=%.8f, h₃=%.8f, v_ee≈%.8f, total≈%.8f\n" h1 h3 soft_coulomb(1.0) (h1+h3+soft_coulomb(1.0))
end

trace_contraction()
