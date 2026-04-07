using ITensors
using ITensorMPS

include("sog_coeffs.jl")

soft_coulomb(u) = 1.0 / sqrt(1.0 + u^2)

"""
    build_hamiltonian_sog(L, delta, rc, z, r; N=2, Ntarget=N, mu_penalty=0)

Build the full Hamiltonian MPO directly using W-matrix construction.
SOG decomposition for e-e interaction: O(L·K) instead of O(L²).
Uses dense (non-QN) tensors. Optionally adds μ*(Ntot-Ntarget)² penalty.

The penalty adds one extra bond channel (constant-range Ntot propagator)
plus diagonal corrections at each site.
"""
function build_hamiltonian_sog(L::Int, delta::Float64, rc::Real, z::Int,
                                r::Vector{Float64}; N::Int=2,
                                Ntarget::Int=N, mu_penalty::Float64=0.0,
                                sites=nothing)
    if sites === nothing
        sites = siteinds("Electron", L; conserve_qns=false)
    end

    K_r = (N > 1) ? N_SOG_REAL : 0
    K_p = (N > 1) ? N_SOG_PAIR : 0
    has_penalty = (mu_penalty > 0.0)
    K_pen = has_penalty ? 1 : 0  # one extra channel for Ntot penalty propagator

    # Bond layout (lower-triangular):
    # 1: LID, 2..5: hopping, 6..5+K_r: real SOG, 6+K_r..5+K_r+2K_p: pair SOG,
    # 5+K_r+2K_p+1: penalty (optional), D: RID
    POS_LID    = 1
    POS_CDAGUP = 2
    POS_CUP    = 3
    POS_CDAGDN = 4
    POS_CDN    = 5
    pos_real(k) = 5 + k
    pos_pair_a(k) = 5 + K_r + 2*(k-1) + 1
    pos_pair_b(k) = pos_pair_a(k) + 1
    POS_PEN    = 5 + K_r + 2*K_p + 1  # penalty channel (if active)
    D = 5 + K_r + 2*K_p + K_pen + 1   # +1 for RID
    POS_RID    = D

    # SOG precomputation
    real_lambda = [exp(-SOG_REAL_S[k] * delta) for k in 1:K_r]
    pair_rk     = [exp(-SOG_PAIR_ALPHA[k] * delta) for k in 1:K_p]
    pair_theta  = [SOG_PAIR_BETA[k] * delta         for k in 1:K_p]

    t_hop = -1.0 / (2 * delta^2)

    links = [Index(D; tags="Link,l=$l") for l in 1:L-1]

    H = MPO(L)
    for l in 1:L
        s = sites[l]
        is_first = (l == 1)
        is_last  = (l == L)

        # Construct W tensor (non-QN, dense)
        if is_first && is_last
            W = ITensor(s', dag(s))
        elseif is_first
            W = ITensor(s', dag(s), links[1])
        elseif is_last
            W = ITensor(s', dag(s), links[L-1])
        else
            W = ITensor(s', dag(s), links[l-1], links[l])
        end

        # Helper: set W element at physical (bra,ket) and bond (a,b)
        function set_welem!(a::Int, b::Int, val::Float64, bra::Int, ket::Int)
            if val == 0.0; return; end
            if is_first && is_last
                W[s' => bra, dag(s) => ket] += val
            elseif is_first
                W[s' => bra, dag(s) => ket, links[1] => b] += val
            elseif is_last
                W[s' => bra, dag(s) => ket, links[L-1] => a] += val
            else
                W[s' => bra, dag(s) => ket, links[l-1] => a, links[l] => b] += val
            end
        end

        # Helper: add coeff * diagonal operator at bond (a,b)
        function add_diag!(a::Int, b::Int, coeff::Float64, diag::Vector{Float64})
            for d in 1:4
                set_welem!(a, b, coeff * diag[d], d, d)
            end
        end

        # Helper: add coeff * off-diagonal operator at bond (a,b)
        function add_offdiag!(a::Int, b::Int, coeff::Float64,
                              elems::Vector{Tuple{Int,Int,Float64}})
            for (bra, ket, v) in elems
                set_welem!(a, b, coeff * v, bra, ket)
            end
        end

        # Diagonal operators
        ID   = [1.0, 1.0, 1.0, 1.0]
        NUP  = [0.0, 1.0, 0.0, 1.0]
        NDN  = [0.0, 0.0, 1.0, 1.0]
        NTOT = [0.0, 1.0, 1.0, 2.0]
        NUPDN= [0.0, 0.0, 0.0, 1.0]

        # Off-diagonal operators: (bra, ket, val)
        CDAGUP = [(2,1,1.0), (4,3,1.0)]
        CUP    = [(1,2,1.0), (3,4,1.0)]
        CDAGDN = [(3,1,1.0), (4,2,-1.0)]
        CDN    = [(1,3,1.0), (2,4,-1.0)]

        # === Diagonal energy: left-id → right-id ===
        h_ke = 1.0 / delta^2
        v_en = -z * soft_coulomb(abs(r[l]))
        add_diag!(POS_LID, POS_RID, h_ke, NUP)
        add_diag!(POS_LID, POS_RID, h_ke, NDN)
        add_diag!(POS_LID, POS_RID, v_en, NUP)
        add_diag!(POS_LID, POS_RID, v_en, NDN)
        if N > 1
            add_diag!(POS_LID, POS_RID, soft_coulomb(0.0), NUPDN)
        end

        # === Identity propagation ===
        if !is_last
            add_diag!(POS_LID, POS_LID, 1.0, ID)
        end
        if !is_first
            add_diag!(POS_RID, POS_RID, 1.0, ID)
        end

        # === Hopping ===
        if !is_last
            add_offdiag!(POS_LID, POS_CDAGUP, 1.0, CDAGUP)
            add_offdiag!(POS_LID, POS_CUP,    1.0, CUP)
            add_offdiag!(POS_LID, POS_CDAGDN, 1.0, CDAGDN)
            add_offdiag!(POS_LID, POS_CDN,    1.0, CDN)
        end
        if !is_first
            add_offdiag!(POS_CDAGUP, POS_RID, t_hop, CUP)
            add_offdiag!(POS_CUP,    POS_RID, t_hop, CDAGUP)
            add_offdiag!(POS_CDAGDN, POS_RID, t_hop, CDN)
            add_offdiag!(POS_CDN,    POS_RID, t_hop, CDAGDN)
        end

        # === SOG e-e interaction ===
        if N > 1
            for k in 1:K_r
                idx = pos_real(k)
                lam = real_lambda[k]
                wt  = SOG_REAL_M[k]
                if !is_last;                add_diag!(POS_LID, idx, lam, NTOT); end
                if !is_first && !is_last;   add_diag!(idx, idx, lam, ID);       end
                if !is_first;               add_diag!(idx, POS_RID, wt, NTOT);  end
            end

            for k in 1:K_p
                ia = pos_pair_a(k)
                ib = pos_pair_b(k)
                rk = pair_rk[k]
                ct = cos(pair_theta[k])
                st = sin(pair_theta[k])
                pk = SOG_PAIR_P[k]
                qk = SOG_PAIR_Q[k]

                if !is_last
                    add_diag!(POS_LID, ia, rk * ct, NTOT)
                    add_diag!(POS_LID, ib, rk * st, NTOT)
                end
                if !is_first && !is_last
                    # Transposed rotation: left-multiplication v^T×M gives forward rotation
                    add_diag!(ia, ia,  rk * ct, ID)
                    add_diag!(ia, ib,  rk * st, ID)
                    add_diag!(ib, ia, -rk * st, ID)
                    add_diag!(ib, ib,  rk * ct, ID)
                end
                if !is_first
                    add_diag!(ia, POS_RID, 2.0 * pk, NTOT)
                    add_diag!(ib, POS_RID, 2.0 * qk, NTOT)
                end
            end
        end

        # === Particle number penalty: μ*(Ntot-Ntarget)² ===
        if has_penalty
            mu = mu_penalty
            Nt = Float64(Ntarget)
            # On-site: μ*[(1-2Nt)*Ntot + 2*Nupdn + Nt²] (Ntot²_onsite = Ntot + 2*Nupdn)
            pen_ntot = mu * (1.0 - 2.0*Nt)
            pen_nupdn = mu * 2.0
            pen_const = mu * Nt^2
            add_diag!(POS_LID, POS_RID, pen_ntot, NTOT)
            add_diag!(POS_LID, POS_RID, pen_nupdn, NUPDN)
            # Constant term (add to identity diagonal)
            if is_first || is_last
                add_diag!(POS_LID, POS_RID, pen_const / L, ID)  # spread across sites
            else
                add_diag!(POS_LID, POS_RID, pen_const / L, ID)
            end
            # Inter-site: 2μ * Σ_{i<j} Ntot_i Ntot_j (constant-range propagator)
            if !is_last;                add_diag!(POS_LID, POS_PEN, 1.0, NTOT);       end
            if !is_first && !is_last;   add_diag!(POS_PEN, POS_PEN, 1.0, ID);         end
            if !is_first;               add_diag!(POS_PEN, POS_RID, 2.0*mu, NTOT);    end
        end

        H[l] = W
    end

    return H, sites
end
