using ITensors, ITensorMPS, Printf, LinearAlgebra

include(joinpath(@__DIR__, "..", "sog", "soe_mpo.jl"))

function main()
    L = 3; rc = 1.5; delta = 1.0; z = 1
    r = [-rc + (i-1)*delta for i in 1:L]

    # QN-conserving SOE W-matrix
    H_w, sites = build_hamiltonian_soe(L, delta, rc, z, r; N=2)
    @printf "W-matrix bond dim = %d\n" maxlinkdim(H_w)

    # QN-conserving OpSum (reference)
    os = OpSum()
    for i in 1:L; os += 1.0/delta^2, "Nup", i; os += 1.0/delta^2, "Ndn", i; end
    for i in 1:L-1
        os += -1.0/(2*delta^2), "Cdagup", i, "Cup", i+1
        os += -1.0/(2*delta^2), "Cdagup", i+1, "Cup", i
        os += -1.0/(2*delta^2), "Cdagdn", i, "Cdn", i+1
        os += -1.0/(2*delta^2), "Cdagdn", i+1, "Cdn", i
    end
    for i in 1:L; v = -z*soft_coulomb(abs(r[i])); os += v, "Nup", i; os += v, "Ndn", i; end
    for i in 1:L; os += soft_coulomb(0.0), "Nupdn", i; end
    for i in 1:L, j in i+1:L
        v = soft_coulomb_soe(abs(r[i]-r[j]))
        os += v, "Nup", i, "Nup", j; os += v, "Ndn", i, "Ndn", j
        os += v, "Nup", i, "Ndn", j; os += v, "Ndn", i, "Nup", j
    end
    H_op = MPO(os, sites)
    @printf "OpSum bond dim = %d\n" maxlinkdim(H_op)

    # Compare via ⟨bra|H|ket⟩ for all 2-particle product states
    states_str = ["Emp", "Up", "Dn", "UpDn"]
    states_code = ["E", "U", "D", "X"]

    @printf "\nDiffering ⟨bra|H|ket⟩ (product states):\n"
    ndiff = 0
    for b1 in 1:4, b2 in 1:4, b3 in 1:4
        for k1 in 1:4, k2 in 1:4, k3 in 1:4
            bra_st = [states_str[b1], states_str[b2], states_str[b3]]
            ket_st = [states_str[k1], states_str[k2], states_str[k3]]
            bra = MPS(sites, bra_st)
            ket = MPS(sites, ket_st)
            e_op = real(inner(bra, Apply(H_op, ket)))
            e_w = real(inner(bra, Apply(H_w, ket)))
            if abs(e_op - e_w) > 1e-10
                ndiff += 1
                bra_s = join([states_code[b1], states_code[b2], states_code[b3]])
                ket_s = join([states_code[k1], states_code[k2], states_code[k3]])
                @printf "  ⟨%s|H|%s⟩: Op=%+.6f  W=%+.6f  diff=%+.6f\n" bra_s ket_s e_op e_w (e_w-e_op)
                if ndiff > 20; println("  ... (truncated)"); return; end
            end
        end
    end
    @printf "\nTotal differing elements: %d\n" ndiff

    if ndiff == 0
        # Run DMRG comparison
        state = fill("Emp", L); state[1] = "Up"; state[2] = "Dn"
        psi0 = MPS(sites, state)
        sweeps = Sweeps(10)
        setmaxdim!(sweeps, fill(50, 10)...)
        setcutoff!(sweeps, fill(1e-12, 10)...)
        setnoise!(sweeps, 1e-5, 1e-6, 1e-7, 1e-8, fill(0.0, 6)...)
        E_w, _ = dmrg(H_w, psi0, sweeps; outputlevel=0)
        E_op, _ = dmrg(H_op, psi0, sweeps; outputlevel=0)
        @printf "\nDMRG: E_w=%.10f  E_op=%.10f  diff=%.2e\n" E_w E_op abs(E_w-E_op)
    end
end

main()
