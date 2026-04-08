using ITensors
using ITensorMPS

include("soe_coeffs.jl")

soft_coulomb(u) = 1.0 / sqrt(1.0 + u^2)

"""
    build_hamiltonian_soe(L, delta, rc, z, r; N=2, sites=nothing)

Build the full Hamiltonian MPO using W-matrix construction with SOE
decomposition for e-e interaction. Construction is O(L·K).

Uses QN-conserving bond indices. Bond dimension D = 6 + K.
"""
function build_hamiltonian_soe(L::Int, delta::Float64, rc::Real, z::Int,
                                r::Vector{Float64}; N::Int=2,
                                sites=nothing)
    if sites === nothing
        sites = siteinds("Electron", L; conserve_qns=true)
    end

    K = (N > 1) ? N_SOE : 0

    POS_LID    = 1
    POS_CDAGUP = 2
    POS_CUP    = 3
    POS_CDAGDN = 4
    POS_CDN    = 5
    pos_soe(k) = 5 + k
    D = 5 + K + 1
    POS_RID    = D

    lambda_k = [exp(-SOE_S[k] * delta) for k in 1:K]
    t_hop = -1.0 / (2 * delta^2)

    # Bond QNs: link carries NEGATED operator flux
    # Diagonal ops (I, Ntot, Nupdn): flux=0 → bond QN(0,0)
    # Cdagup: flux=(+1,+1) → bond QN(-1,-1)
    # Cup:    flux=(-1,-1) → bond QN(+1,+1)
    # Cdagdn: flux=(+1,-1) → bond QN(-1,+1)
    # Cdn:    flux=(-1,+1) → bond QN(+1,-1)
    qn0    = QN(("Nf",  0, -1), ("Sz",  0))
    qnCdU  = QN(("Nf", -1, -1), ("Sz", -1))  # bond for Cdagup
    qncU   = QN(("Nf",  1, -1), ("Sz",  1))   # bond for Cup
    qnCdD  = QN(("Nf", -1, -1), ("Sz",  1))   # bond for Cdagdn
    qncD   = QN(("Nf",  1, -1), ("Sz", -1))   # bond for Cdn

    # qn0 block: LID(1) + SOE(K) + RID(1) = K+2 states
    function make_link(l)
        return Index(
            qn0   => K + 2,
            qnCdU => 1,
            qncU  => 1,
            qnCdD => 1,
            qncD  => 1;
            tags="Link,l=$l"
        )
    end

    links = [make_link(l) for l in 1:L-1]

    # Flat index within the QN Index for each logical slot
    # Block order: qn0 (K+2), qnCdU (1), qncU (1), qnCdD (1), qncD (1)
    # qn0 block positions: LID=1, SOE_1=2, ..., SOE_K=K+1, RID=K+2
    function flat_idx(slot)
        if slot == POS_LID;     return 1
        elseif slot == POS_RID; return K + 2
        elseif 5 < slot <= 5+K; return slot - 4  # SOE k → position k+1
        elseif slot == POS_CDAGUP; return K + 3
        elseif slot == POS_CUP;    return K + 4
        elseif slot == POS_CDAGDN; return K + 5
        elseif slot == POS_CDN;    return K + 6
        end
    end

    H = MPO(L)
    for l in 1:L
        s = sites[l]
        is_first = (l == 1)
        is_last  = (l == L)

        if is_first && is_last
            W = ITensor(s', dag(s))
        elseif is_first
            W = ITensor(s', dag(s), links[1])
        elseif is_last
            W = ITensor(dag(links[L-1]), s', dag(s))
        else
            W = ITensor(dag(links[l-1]), s', dag(s), links[l])
        end

        function set_welem!(a_slot::Int, b_slot::Int, val::Float64, bra::Int, ket::Int)
            val == 0.0 && return
            if is_first && is_last
                W[s' => bra, dag(s) => ket] += val
            elseif is_first
                W[s' => bra, dag(s) => ket, links[1] => flat_idx(b_slot)] += val
            elseif is_last
                W[dag(links[L-1]) => flat_idx(a_slot), s' => bra, dag(s) => ket] += val
            else
                W[dag(links[l-1]) => flat_idx(a_slot), s' => bra, dag(s) => ket,
                  links[l] => flat_idx(b_slot)] += val
            end
        end

        function add_diag!(a::Int, b::Int, coeff::Float64, diag::Vector{Float64})
            for d in 1:4
                set_welem!(a, b, coeff * diag[d], d, d)
            end
        end

        function add_offdiag!(a::Int, b::Int, coeff::Float64,
                              elems::Vector{Tuple{Int,Int,Float64}})
            for (bra, ket, v) in elems
                set_welem!(a, b, coeff * v, bra, ket)
            end
        end

        ID   = [1.0, 1.0, 1.0, 1.0]
        NUP  = [0.0, 1.0, 0.0, 1.0]
        NDN  = [0.0, 0.0, 1.0, 1.0]
        NTOT = [0.0, 1.0, 1.0, 2.0]
        NUPDN= [0.0, 0.0, 0.0, 1.0]

        F    = [1.0, -1.0, -1.0, 1.0]  # (-1)^(nup+ndn) fermion parity

        CDAGUP = [(2,1,1.0), (4,3,1.0)]
        CUP    = [(1,2,1.0), (3,4,1.0)]
        CDAGDN = [(3,1,1.0), (4,2,-1.0)]
        CDN    = [(1,3,1.0), (2,4,-1.0)]

        # On-site diagonal
        h_ke = 1.0 / delta^2
        v_en = -z * soft_coulomb(abs(r[l]))
        add_diag!(POS_LID, POS_RID, h_ke, NUP)
        add_diag!(POS_LID, POS_RID, h_ke, NDN)
        add_diag!(POS_LID, POS_RID, v_en, NUP)
        add_diag!(POS_LID, POS_RID, v_en, NDN)
        if N > 1
            add_diag!(POS_LID, POS_RID, soft_coulomb(0.0), NUPDN)
        end

        # Identity propagation
        if !is_last;  add_diag!(POS_LID, POS_LID, 1.0, ID); end
        if !is_first; add_diag!(POS_RID, POS_RID, 1.0, ID); end

        # Hopping: c†_{i,σ} c_{j,σ} requires Jordan-Wigner string
        # In JW encoding: c†_i c_j = (c†_i F_i) * F_{i+1} * ... * F_{j-1} * c_j
        # W-matrix:
        #   Start: op * F (operator times fermion parity at source site)
        #   Propagate: F on intermediate sites
        #   End: bare operator at sink site

        # op*F means: for off-diagonal (bra, ket, v), multiply v by F[ket]
        # since the matrix element becomes ⟨bra|op*F|ket⟩ = ⟨bra|op|ket⟩ * F[ket]... no.
        # op*F is the matrix product: (op*F)[bra,ket] = op[bra, k] * F[k, ket]
        # Since F is diagonal: (op*F)[bra,ket] = op[bra,ket] * F[ket]
        CDAGUP_F = [(2,1, 1.0*F[1]), (4,3, 1.0*F[3])]
        CUP_F    = [(1,2, 1.0*F[2]), (3,4, 1.0*F[4])]
        CDAGDN_F = [(3,1, 1.0*F[1]), (4,2,-1.0*F[2])]
        CDN_F    = [(1,3, 1.0*F[3]), (2,4,-1.0*F[4])]

        if !is_last
            add_offdiag!(POS_LID, POS_CDAGUP, 1.0, CDAGUP_F)
            add_offdiag!(POS_LID, POS_CUP,    1.0, CUP_F)
            add_offdiag!(POS_LID, POS_CDAGDN, 1.0, CDAGDN_F)
            add_offdiag!(POS_LID, POS_CDN,    1.0, CDN_F)
        end
        # No propagation: nearest-neighbor hopping only (start → end on adjacent sites)
        # End hopping channels:
        #   CDAGUP→RID: t_hop*Cup    (represents t * c†_i c_{i+1})
        #   CUP→RID:   -t_hop*Cdagup (represents t * c†_{i+1} c_i = -t * c_i c†_{i+1})
        #   The sign flip accounts for fermionic anticommutation when
        #   reordering c†_{i+1} c_i → -c_i c†_{i+1} for left-to-right MPO
        if !is_first
            add_offdiag!(POS_CDAGUP, POS_RID,  t_hop, CUP)
            add_offdiag!(POS_CUP,    POS_RID, -t_hop, CDAGUP)
            add_offdiag!(POS_CDAGDN, POS_RID,  t_hop, CDN)
            add_offdiag!(POS_CDN,    POS_RID, -t_hop, CDAGDN)
        end

        # SOE e-e interaction
        if N > 1
            for k in 1:K
                idx = pos_soe(k)
                lam = lambda_k[k]
                wt  = SOE_W[k]
                if !is_last;                add_diag!(POS_LID, idx, lam, NTOT); end
                if !is_first && !is_last;   add_diag!(idx, idx, lam, ID);       end
                if !is_first;               add_diag!(idx, POS_RID, wt, NTOT);  end
            end
        end

        H[l] = W
    end

    return H, sites
end
