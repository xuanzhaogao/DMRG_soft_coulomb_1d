using ITensors
using ITensorMPS
using HDF5
using Printf

include(joinpath(@__DIR__, "..", "sog", "soe_mpo.jl"))

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

function run_dmrg_converge(H, psi0; max_bond::Int=1000, cutoff::Float64=1e-12,
                            etol::Float64=1e-8, max_sweeps::Int=200,
                            sweeps_per_batch::Int=5)
    ramp = filter(<=(max_bond), [50, 100, 200, 400, 600, 800, 1000])
    n_ramp = length(ramp)
    noise_ramp = [1e-5, 1e-6, 1e-7, 1e-8]
    n_ramp_total = max(n_ramp, length(noise_ramp))
    dims_ramp = vcat(ramp, fill(max_bond, max(n_ramp_total - n_ramp, 0)))[1:n_ramp_total]
    noise_full = vcat(noise_ramp, fill(0.0, max(n_ramp_total - length(noise_ramp), 0)))[1:n_ramp_total]

    sw = Sweeps(n_ramp_total)
    setmaxdim!(sw, dims_ramp...)
    setcutoff!(sw, fill(cutoff, n_ramp_total)...)
    setnoise!(sw, noise_full...)

    @info "Phase 1: ramp-up ($n_ramp_total sweeps)"
    energy, psi = dmrg(H, psi0, sw; outputlevel=1)
    total = n_ramp_total
    @printf "  After ramp-up (%d sweeps): E = %.13f\n" total energy
    flush(stdout)

    batch = Sweeps(sweeps_per_batch)
    setmaxdim!(batch, fill(max_bond, sweeps_per_batch)...)
    setcutoff!(batch, fill(cutoff, sweeps_per_batch)...)
    setnoise!(batch, fill(0.0, sweeps_per_batch)...)

    prev = energy
    while total < max_sweeps
        energy, psi = dmrg(H, psi, batch; outputlevel=1)
        total += sweeps_per_batch
        dE = abs(energy - prev)
        @printf "  After %d sweeps: E = %.13f  |ΔE| = %.3e\n" total energy dE
        flush(stdout)
        if dE < etol
            @info "Converged: |ΔE| = $dE < $etol after $total sweeps"
            break
        end
        prev = energy
    end
    if total >= max_sweeps
        @warn "Max sweeps reached ($max_sweeps), |ΔE| = $(abs(energy - prev))"
    end
    return energy, psi
end

function main()
    systems = Dict(
        "Be" => (N=4, z=4, Nup=2, Ndn=2, rc=20),
        "B"  => (N=5, z=5, Nup=3, Ndn=2, rc=20),
        "C"  => (N=6, z=6, Nup=3, Ndn=3, rc=20),
        "N"  => (N=7, z=7, Nup=4, Ndn=3, rc=20),
        "O"  => (N=8, z=8, Nup=4, Ndn=4, rc=20),
    )
    refs = Dict(
        "Be" => -6.78507795170852,
        "B"  => -9.81971246294425,
        "C"  => -13.33154314459545,
        "N"  => -17.29164709368697,
        "O"  => -21.69957056180725,
    )

    name = get(ENV, "DMRG_SYSTEM", "Be")
    delta = 0.1
    max_bond = parse(Int, get(ENV, "DMRG_MAXBOND", "1000"))
    max_sweeps = parse(Int, get(ENV, "DMRG_MAXSWEEPS", "200"))
    etol = parse(Float64, get(ENV, "DMRG_ETOL", "1e-8"))

    if !haskey(systems, name)
        error("Unknown system: $name. Available: $(join(keys(systems), ", "))")
    end

    sys = systems[name]
    rc = sys.rc
    L = round(Int, 2*rc / delta)
    r = [-rc + (i-1)*delta for i in 1:L]

    @printf "\n%s\n" repeat("=", 60)
    @printf "System: %s  N=%d  z=%d  Nup=%d  Ndn=%d\n" name sys.N sys.z sys.Nup sys.Ndn
    @printf "Grid:   rc=%d  L=%d  δ=%.4f  max_bond=%d\n" rc L delta max_bond
    @printf "%s\n" repeat("=", 60)
    flush(stdout)

    @printf "[SOE] Building MPO... "
    flush(stdout)
    t_build = @elapsed (H, sites) = build_hamiltonian_soe(L, delta, rc, sys.z, r; N=sys.N)
    @printf "%.2f s (bond dim=%d, K=%d)\n" t_build maxlinkdim(H) N_SOE
    flush(stdout)

    psi0 = make_initial_mps(sites, L, sys.Nup, sys.Ndn)

    @time energy, psi = run_dmrg_converge(H, psi0;
        max_bond=max_bond, etol=etol, max_sweeps=max_sweeps)

    ref = get(refs, name, NaN)
    err = energy - ref
    @printf "\n%s  E = %.13f   ref = %.13f   err = %+.3e\n" name energy ref err
    flush(stdout)

    mkpath("data")
    h5open("data/$(name)_soe_psi.h5", "w") do f
        write(f, "psi", psi)
        write(f, "energy", energy)
    end
end

main()
