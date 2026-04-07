include("sog_mpo.jl")
using Printf

function test_all()
    for (L, rc) in [(2, 0.5), (3, 0.75), (4, 1.0), (6, 1.5)]
        delta = 0.5; z = 2; N = 2
        r = [-rc + (i-1)*delta for i in 1:L]

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
            os += v_ee, "Nup", i, "Nup", j
            os += v_ee, "Ndn", i, "Ndn", j
            os += v_ee, "Nup", i, "Ndn", j
            os += v_ee, "Ndn", i, "Nup", j
        end
        H_n = MPO(os, sites_n)
        H_s, sites_s = build_hamiltonian_sog(L, delta, rc, z, r; N=N, conserve_qns=false)

        max_err = 0.0
        for i in 1:L, j in i+1:L
            st = fill("Emp", L); st[i] = "Up"; st[j] = "Dn"
            En = inner(MPS(sites_n, st)', H_n, MPS(sites_n, st))
            Es = inner(MPS(sites_s, copy(st))', H_s, MPS(sites_s, copy(st)))
            max_err = max(max_err, abs(Es - En))
        end
        @printf "L=%d: max |diff| across all pairs = %.2e\n" L max_err
    end
end

test_all()
