using ITensors
using ITensorMPS
using HDF5
using Printf
using CUDA

# ============================================================
# 1D soft-Coulomb DMRG with SOE-MPO + GPU support
# Uses O(L·K) SOE-MPO instead of O(L²) OpSum MPO
# ============================================================

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

function run_dmrg(H, psi0; max_bond::Int=500, cutoff::Float64=1e-12,
                   use_gpu::Bool=false, etol::Float64=1e-8, max_sweeps::Int=200,
                   sweeps_per_batch::Int=5)
    # Phase 1: ramp-up sweeps to grow bond dimension
    ramp = filter(<=(max_bond), [50, 100, 200, 400, 600, 800, 1000])
    n_ramp = length(ramp)
    noise_ramp = [1e-5, 1e-6, 1e-7, 1e-8]
    n_ramp_total = max(n_ramp, length(noise_ramp))
    dims_ramp = vcat(ramp, fill(max_bond, max(n_ramp_total - n_ramp, 0)))[1:n_ramp_total]
    noise_ramp_full = vcat(noise_ramp, fill(0.0, max(n_ramp_total - length(noise_ramp), 0)))[1:n_ramp_total]

    sweeps_ramp = Sweeps(n_ramp_total)
    setmaxdim!(sweeps_ramp, dims_ramp...)
    setcutoff!(sweeps_ramp, fill(cutoff, n_ramp_total)...)
    setnoise!(sweeps_ramp, noise_ramp_full...)

    if use_gpu
        @info "Moving H and psi to GPU..."
        H_run = cu(H)
        psi_run = cu(psi0)
    else
        H_run = H
        psi_run = psi0
    end

    @info "Phase 1: ramp-up ($n_ramp_total sweeps)"
    energy, psi_run = dmrg(H_run, psi_run, sweeps_ramp; outputlevel=1)
    total_sweeps = n_ramp_total
    @printf "  After ramp-up (%d sweeps): E = %.13f\n" total_sweeps energy
    flush(stdout)

    # Phase 2: iterate at full bond dimension until convergence
    batch_sweeps = Sweeps(sweeps_per_batch)
    setmaxdim!(batch_sweeps, fill(max_bond, sweeps_per_batch)...)
    setcutoff!(batch_sweeps, fill(cutoff, sweeps_per_batch)...)
    setnoise!(batch_sweeps, fill(0.0, sweeps_per_batch)...)

    prev_energy = energy
    while total_sweeps < max_sweeps
        energy, psi_run = dmrg(H_run, psi_run, batch_sweeps; outputlevel=1)
        total_sweeps += sweeps_per_batch
        dE = abs(energy - prev_energy)
        @printf "  After %d sweeps: E = %.13f  |ΔE| = %.3e\n" total_sweeps energy dE
        flush(stdout)
        if dE < etol
            @info "Energy converged: |ΔE| = $(dE) < $(etol) after $total_sweeps sweeps"
            break
        end
        prev_energy = energy
    end
    if total_sweeps >= max_sweeps
        @warn "Reached max_sweeps=$max_sweeps without converging (|ΔE| = $(abs(energy - prev_energy)))"
    end

    psi_cpu = use_gpu ? NDTensors.cpu(psi_run) : psi_run
    return energy, psi_cpu
end

function main()
    systems = [
        ("H",  1, 1, 1, 0, 15),
        ("He", 2, 2, 1, 1, 13),
        ("Li", 3, 3, 2, 1, 20),
        ("Be", 4, 4, 2, 2, 20),
        ("B",  5, 5, 3, 2, 20),
        ("C",  6, 6, 3, 3, 20),
        ("N",  7, 7, 4, 3, 20),
        ("O",  8, 8, 4, 4, 20),
    ]
    refs = Dict(
        "H"  => -0.6697771382138,
        "He" => -2.2382578241080,
        "Li" => -4.210531647613249,
        "Be" => -6.78507795170852,
        "B"  => -9.81971246294425,
        "C"  => -13.33154314459545,
        "N"  => -17.29164709368697,
        "O"  => -21.69957056180725,
    )

    delta = 0.1
    run_list   = Set(strip.(split(get(ENV, "DMRG_SYSTEMS", "H,He"), ",")))
    max_bond   = parse(Int, get(ENV, "DMRG_MAXBOND", "500"))
    max_sweeps = parse(Int, get(ENV, "DMRG_MAXSWEEPS", "200"))
    etol       = parse(Float64, get(ENV, "DMRG_ETOL", "1e-8"))
    use_gpu    = parse(Bool, get(ENV, "USE_GPU", "false"))

    if use_gpu && CUDA.functional()
        @info "GPU: $(CUDA.name(CUDA.device())), $(CUDA.totalmem(CUDA.device()) ÷ 1024^2) MiB"
    elseif use_gpu
        @warn "USE_GPU=true but CUDA not functional — falling back to CPU"
        use_gpu = false
    end

    results = Dict{String, Float64}()

    for (name, N, z, Nup, Ndn, rc) in systems
        name ∉ run_list && continue
        L = round(Int, 2*rc / delta)
        r = [-rc + (i-1)*delta for i in 1:L]

        @printf "\n%s\n" repeat("=", 60)
        @printf "System: %s  N=%d  z=%d  Nup=%d  Ndn=%d\n" name N z Nup Ndn
        @printf "Grid:   rc=%.0f  L=%d  δ=%.4f\n" rc L delta
        @printf "%s\n" repeat("=", 60)
        flush(stdout)

        @time begin
            H, sites = build_hamiltonian_soe(L, delta, rc, z, r; N=N)
            @printf "MPO bond dim = %d  (SOE kernel, %d terms)\n" maxlinkdim(H) N_SOE
            flush(stdout)

            psi0 = make_initial_mps(sites, L, Nup, Ndn)
            energy, psi = run_dmrg(H, psi0; max_bond=max_bond, cutoff=1e-12,
                                    use_gpu=use_gpu, etol=etol, max_sweeps=max_sweeps)
        end

        results[name] = energy
        ref = get(refs, name, NaN)
        err = energy - ref
        @printf "\n%-4s  E = %.13f   ref = %.13f   err = %+.3e\n" name energy ref err
        flush(stdout)

        mkpath("data")
        h5open("data/$(name)_soe_psi.h5", "w") do f
            write(f, "psi", psi)
            write(f, "energy", energy)
        end
    end

    @printf "\n%s\n" repeat("=", 60)
    @printf "%-6s  %-20s  %-20s  %-12s\n" "System" "E_DMRG" "E_ref" "Error"
    @printf "%s\n" repeat("-", 60)
    for (name, energy) in sort(collect(results))
        ref = get(refs, name, NaN)
        @printf "%-6s  %20.13f  %20.13f  %+12.3e\n" name energy ref (energy-ref)
    end
end

main()
