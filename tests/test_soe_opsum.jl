using ITensors, ITensorMPS, Printf

include(joinpath(@__DIR__, "..", "sog", "soe_coeffs.jl"))

soft_coulomb(u) = 1.0 / sqrt(1.0 + u^2)

function main()
    rc = 3.0; delta = 0.5; z = 2; N = 2; Nup = 1; Ndn = 1
    L = round(Int, 2*rc / delta)
    r = [-rc + (i-1)*delta for i in 1:L]

    sites = siteinds("Electron", L; conserve_qns=true)
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
    H = MPO(os, sites)
    @printf "SOE-OpSum MPO bond dim = %d\n" maxlinkdim(H)

    state = fill("Emp", L)
    state[4] = "Up"; state[5] = "Dn"
    psi0 = MPS(sites, state)

    sweeps = Sweeps(10)
    setmaxdim!(sweeps, fill(50, 10)...)
    setcutoff!(sweeps, fill(1e-10, 10)...)
    setnoise!(sweeps, 1e-5, 1e-6, 1e-7, 1e-8, 0, 0, 0, 0, 0, 0)

    E, _ = dmrg(H, psi0, sweeps; outputlevel=1)
    @printf "\nSOE-OpSum E = %.10f\n" E
    @printf "Naive ref E = %.10f\n" -2.2266547408
    @printf "ΔE = %.2e\n" abs(E - (-2.2266547408))
end

main()
