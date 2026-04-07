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

function test_penalty()
    rc = 3.0; delta = 0.5; z = 2; N = 2; Nup = 1; Ndn = 1
    L = round(Int, 2*rc / delta)
    r = [-rc + (i-1)*delta for i in 1:L]

    @printf "He: rc=%.1f, δ=%.1f, L=%d\n\n" rc delta L

    mu = 100.0  # penalty strength
    H, sites = build_hamiltonian_sog(L, delta, rc, z, r; N=N, Ntarget=N, mu_penalty=mu)
    @printf "SOG+penalty MPO: bond dim=%d, μ=%.0f\n" maxlinkdim(H) mu

    psi0 = make_initial_mps(sites, L, Nup, Ndn)
    nsweeps = 20
    maxdim = vcat([50, 100, 200, 400], fill(500, nsweeps-4))
    noise = vcat([1e-4, 1e-5, 1e-6, 1e-7, 1e-8], fill(0.0, nsweeps-5))
    sweeps = Sweeps(nsweeps)
    setmaxdim!(sweeps, maxdim[1:nsweeps]...)
    setcutoff!(sweeps, fill(1e-12, nsweeps)...)
    setnoise!(sweeps, noise[1:nsweeps]...)

    E, psi = dmrg(H, psi0, sweeps; outputlevel=1)
    ntot = sum(expect(psi, "Ntot"))
    @printf "\nE_total = %.10f (includes penalty)\n" E
    @printf "⟨Ntot⟩ = %.6f\n" ntot
    # Physical energy = E_total - μ*(⟨Ntot⟩-Ntarget)²
    E_phys = E - mu * (ntot - N)^2
    @printf "E_phys ≈ %.10f\n" E_phys
    @printf "Reference = -2.2266547408\n"
    @printf "Diff = %+.2e\n" (E_phys - (-2.2266547408))
end

test_penalty()
