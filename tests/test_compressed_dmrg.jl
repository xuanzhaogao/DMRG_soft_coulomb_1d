include("sog_mpo.jl")
using Printf

function spread(n, lo, hi)
    n == 0 && return Int[]
    n == 1 && return [div(lo + hi, 2)]
    round.(Int, range(lo, hi, length=n))
end
function make_initial_mps(sites, L, Nup, Ndn)
    state = fill("Emp", L)
    for i in spread(Nup, 1, L); state[i] = "Up"; end
    cands = setdiff(1:L, spread(Nup, 1, L))
    for i in (isempty(cands) ? spread(Ndn,1,L) : cands[spread(Ndn,1,length(cands))])
        state[i] = state[i] == "Up" ? "UpDn" : "Dn"
    end
    MPS(sites, state)
end

function test()
    for (label, rc, delta_val) in [("small He", 3.0, 0.5), ("full He", 13.0, 0.1)]
        z = 2; N = 2; Nup = 1; Ndn = 1
        L = round(Int, 2*rc / delta_val)
        r = [-rc + (i-1)*delta_val for i in 1:L]

        @printf "\n=== %s (rc=%.1f, δ=%.2f, L=%d) ===\n" label rc delta_val L

        t_build = @elapsed begin
            H_raw, sites = build_hamiltonian_sog(L, delta_val, rc, z, r; N=N, Ntarget=N, mu_penalty=100.0)
            H = truncate(H_raw; cutoff=1e-12)
        end
        @printf "Build: %.2fs, raw D=%d → compressed D=%d\n" t_build maxlinkdim(H_raw) maxlinkdim(H)

        psi0 = make_initial_mps(sites, L, Nup, Ndn)
        nsweeps = 20
        maxdim = vcat([50, 100, 200, 400], fill(500, nsweeps-4))
        noise = vcat([1e-5, 1e-6, 1e-7, 1e-8], fill(0.0, nsweeps-4))
        sweeps = Sweeps(nsweeps)
        setmaxdim!(sweeps, maxdim[1:nsweeps]...)
        setcutoff!(sweeps, fill(1e-12, nsweeps)...)
        setnoise!(sweeps, noise[1:nsweeps]...)

        E, psi = dmrg(H, psi0, sweeps; outputlevel=1)
        ntot = sum(expect(psi, "Ntot"))
        E_phys = E - 100.0 * (ntot - N)^2
        @printf "\nE_total = %.10f  ⟨Ntot⟩=%.4f  E_phys=%.10f\n" E ntot E_phys
    end
end

test()
