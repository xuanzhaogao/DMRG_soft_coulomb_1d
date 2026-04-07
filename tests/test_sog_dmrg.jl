include("sog_mpo.jl")
using Printf

function spread(n::Int, lo::Int, hi::Int)
    n == 0 && return Int[]
    n == 1 && return [div(lo + hi, 2)]
    round.(Int, range(lo, hi, length=n))
end

function make_initial_mps(sites, L, Nup, Ndn)
    state = fill("Emp", L)
    up_pos = spread(Nup, 1, L)
    for i in up_pos; state[i] = "Up"; end
    candidates = setdiff(1:L, up_pos)
    dn_pos = isempty(candidates) ? spread(Ndn, 1, L) :
             candidates[spread(Ndn, 1, length(candidates))]
    for i in dn_pos; state[i] = (state[i] == "Up") ? "UpDn" : "Dn"; end
    return MPS(sites, state)
end

function test_he_sog()
    rc = 3.0; delta = 0.5; z = 2; N = 2; Nup = 1; Ndn = 1
    L = round(Int, 2*rc / delta)
    r = [-rc + (i-1)*delta for i in 1:L]

    @printf "He: rc=%.1f, δ=%.1f, L=%d\n" rc delta L

    t = @elapsed (H, sites) = build_hamiltonian_sog(L, delta, rc, z, r; N=N)
    @printf "SOG-MPO built in %.2fs (bond dim=%d)\n" t maxlinkdim(H)

    psi0 = make_initial_mps(sites, L, Nup, Ndn)

    nsweeps = 20
    maxdim = vcat([50, 100, 200, 400], fill(500, nsweeps - 4))
    noise = vcat([1e-4, 1e-5, 1e-6, 1e-7, 1e-8], fill(0.0, nsweeps - 5))
    sweeps = Sweeps(nsweeps)
    setmaxdim!(sweeps, maxdim[1:nsweeps]...)
    setcutoff!(sweeps, fill(1e-12, nsweeps)...)
    setnoise!(sweeps, noise[1:nsweeps]...)

    E, psi = dmrg(H, psi0, sweeps; outputlevel=1)

    ntot = sum(expect(psi, "Ntot"))
    sz = sum(expect(psi, "Sz"))
    ref = -2.2266547408
    @printf "\nE = %.10f  ref = %.10f  diff = %+.2e\n" E ref (E - ref)
    @printf "⟨Ntot⟩=%.4f  ⟨Sz⟩=%.4f  maxlinkdim=%d\n" ntot sz maxlinkdim(psi)
end

test_he_sog()
