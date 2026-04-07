using ITensors
using ITensorMPS
using Printf

# Bond dimension self-convergence check for H and He
# Runs DMRG at increasing max_bond and tracks energy convergence

function soft_coulomb(u)
    return 1.0 / sqrt(1.0 + u^2)
end

function build_hamiltonian(L::Int, delta::Float64, rc::Real, z::Int,
                            r::Vector{Float64}; N::Int=2)
    sites = siteinds("Electron", L; conserve_qns=true)
    os = OpSum()
    for i in 1:L
        os += 1.0/delta^2, "Nup", i
        os += 1.0/delta^2, "Ndn", i
    end
    for i in 1:L-1
        os += -1.0/(2*delta^2), "Cdagup", i, "Cup",   i+1
        os += -1.0/(2*delta^2), "Cdagup", i+1, "Cup", i
        os += -1.0/(2*delta^2), "Cdagdn", i, "Cdn",   i+1
        os += -1.0/(2*delta^2), "Cdagdn", i+1, "Cdn", i
    end
    for i in 1:L
        v_en = -z * soft_coulomb(abs(r[i]))
        os += v_en, "Nup", i
        os += v_en, "Ndn", i
    end
    if N > 1
        for i in 1:L
            os += soft_coulomb(0.0), "Nupdn", i
        end
        for i in 1:L, j in i+1:L
            v_ee = soft_coulomb(abs(r[i] - r[j]))
            os += v_ee, "Nup", i, "Nup", j
            os += v_ee, "Ndn", i, "Ndn", j
            os += v_ee, "Nup", i, "Ndn", j
            os += v_ee, "Ndn", i, "Nup", j
        end
    end
    H = MPO(os, sites)
    return H, sites
end

function spread(n::Int, lo::Int, hi::Int)
    n == 0 && return Int[]
    n == 1 && return [div(lo + hi, 2)]
    round.(Int, range(lo, hi, length=n))
end

function make_initial_mps(sites, L::Int, Nup::Int, Ndn::Int)
    state = fill("Emp", L)
    up_pos = spread(Nup, 1, L)
    for i in up_pos; state[i] = "Up"; end
    candidates = setdiff(1:L, up_pos)
    dn_pos = isempty(candidates) ? spread(Ndn, 1, L) :
             candidates[spread(Ndn, 1, length(candidates))]
    for i in dn_pos; state[i] = (state[i] == "Up") ? "UpDn" : "Dn"; end
    return MPS(sites, state)
end

function run_dmrg_at_bond(H, sites, Nup, Ndn, L, max_bond; cutoff=1e-12, nsweeps=20)
    ramp = [50, 100, 200, 400, 600, 800, 1000]
    dims = vcat(ramp, fill(max_bond, max(nsweeps - length(ramp), 0)))[1:nsweeps]
    # cap ramp at max_bond
    dims = min.(dims, max_bond)
    noise = vcat([1e-5, 1e-6, 1e-7, 1e-8], fill(0.0, max(nsweeps-4, 0)))[1:nsweeps]

    sweeps = Sweeps(nsweeps)
    setmaxdim!(sweeps, dims...)
    setcutoff!(sweeps, fill(cutoff, nsweeps)...)
    setnoise!(sweeps, noise...)

    psi0 = make_initial_mps(sites, L, Nup, Ndn)
    energy, psi = dmrg(H, psi0, sweeps; outputlevel=0)
    return energy, maxlinkdim(psi)
end

function main()
    systems = [
        ("H",  1, 1, 1, 0, 15),
        ("He", 2, 2, 1, 1, 13),
    ]
    refs = Dict("H" => -0.6697771382138, "He" => -2.2382578241080)

    run_list = Set(strip.(split(get(ENV, "DMRG_SYSTEMS", "H,He"), ",")))
    delta = 0.1
    bond_dims = [50, 100, 200, 500, 1000]

    for (name, N, z, Nup, Ndn, rc) in systems
        name ∉ run_list && continue
        L = round(Int, 2*rc / delta)
        r = [-rc + (i-1)*delta for i in 1:L]

        @printf "\n%s  (L=%d, δ=%.2f)\n" name L delta
        @printf "%s\n" repeat("-", 65)
        @printf "%-8s  %-22s  %-8s  %-12s\n" "max_χ" "Energy" "act_χ" "ΔE(prev)"
        @printf "%s\n" repeat("-", 65)
        flush(stdout)

        H, sites = build_hamiltonian(L, delta, rc, z, r; N=N)

        prev_E = NaN
        for χ in bond_dims
            E, act_χ = run_dmrg_at_bond(H, sites, Nup, Ndn, L, χ)
            dE = isnan(prev_E) ? NaN : E - prev_E
            @printf "%-8d  %22.13f  %-8d  %s\n" χ E act_χ (isnan(dE) ? "         —" : @sprintf("%+.3e", dE))
            flush(stdout)
            prev_E = E
            # converged if energy change < 1e-10
            if !isnan(dE) && abs(dE) < 1e-10
                @printf "  → Converged at χ=%d (ΔE < 1e-10)\n" χ
                break
            end
        end

        ref = refs[name]
        @printf "\n  Best E = %.13f   ref = %.13f   err = %+.3e\n" prev_E ref (prev_E - ref)
        flush(stdout)
    end
end

main()
