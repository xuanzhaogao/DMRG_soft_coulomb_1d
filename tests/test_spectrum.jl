include("sog_mpo.jl")
using Printf, LinearAlgebra

function test_spectrum()
    rc = 0.5; delta = 0.5; z = 2; L = 2; N = 2
    r = [-rc + (i-1)*delta for i in 1:L]

    # Build both Hamiltonians without QN for full matrix comparison
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
    H_s, sites_s = build_hamiltonian_sog(L, delta, rc, z, r; N=N, conserve_qns=false)

    # Convert to full matrices
    d = 4^L  # total Hilbert space dimension
    Hn_full = zeros(d, d)
    Hs_full = zeros(d, d)

    for i in 1:d
        # Create basis vector |i⟩
        ket_n = zeros(d); ket_n[i] = 1.0
        ket_s = zeros(d); ket_s[i] = 1.0

        # Map to MPS
        psi_n = MPS(sites_n, [((i-1) >> (2*(l-1))) % 4 + 1 for l in 1:L])
        # Actually, this is wrong. Let me use a different approach.
    end

    # Simpler: contract MPO into a single tensor
    T_n = H_n[1]
    T_s = H_s[1]
    for l in 2:L
        T_n = T_n * H_n[l]
        T_s = T_s * H_s[l]
    end

    # T should be a 2L-index tensor (L primed, L unprimed site indices)
    # Combine into a matrix
    combo_bra = [sites_n[l]' for l in 1:L]
    combo_ket = [dag(sites_n[l]) for l in 1:L]

    @printf "T_n indices: %d\n" length(inds(T_n))
    @printf "T_s indices: %d\n" length(inds(T_s))

    # Extract full matrix
    Hn = zeros(d, d)
    Hs = zeros(d, d)
    for i in 1:d, j in 1:d
        # Decode multi-index
        ii = i - 1; jj = j - 1
        idx_bra = [(ii >> (2*(l-1))) % 4 + 1 for l in 1:L]
        idx_ket = [(jj >> (2*(l-1))) % 4 + 1 for l in 1:L]

        kw_n = Tuple(vcat([sites_n[l]' => idx_bra[l] for l in 1:L],
                          [dag(sites_n[l]) => idx_ket[l] for l in 1:L]))
        kw_s = Tuple(vcat([sites_s[l]' => idx_bra[l] for l in 1:L],
                          [dag(sites_s[l]) => idx_ket[l] for l in 1:L]))
        Hn[i,j] = T_n[kw_n...]
        Hs[i,j] = T_s[kw_s...]
    end

    # Compare eigenvalues
    en = eigvals(Symmetric(Hn))
    es = eigvals(Symmetric(Hs))

    @printf "\nLowest 10 eigenvalues:\n"
    @printf "%-6s  %15s  %15s  %12s\n" "#" "Naive" "SOG" "Diff"
    for k in 1:min(10, d)
        @printf "%-6d  %15.8f  %15.8f  %+12.2e\n" k en[k] es[k] (es[k]-en[k])
    end

    @printf "\nMax |H_naive - H_sog| = %.2e\n" maximum(abs.(Hn .- Hs))
    @printf "Frobenius norm diff = %.2e\n" norm(Hn .- Hs)
end

test_spectrum()
