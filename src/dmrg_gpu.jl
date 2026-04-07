using ITensors
using ITensorMPS
using HDF5
using Printf
using CUDA

# ============================================================
# 1D soft-Coulomb DMRG with GPU support
# ============================================================

soft_coulomb(u) = 1.0 / sqrt(1.0 + u^2)

function spread(n::Int, lo::Int, hi::Int)
    n == 0 && return Int[]
    n == 1 && return [div(lo + hi, 2)]
    round.(Int, range(lo, hi, length=n))
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
        os += -1.0/(2*delta^2), "Cdagup", i,   "Cup",    i+1
        os += -1.0/(2*delta^2), "Cdagup", i+1, "Cup",    i
        os += -1.0/(2*delta^2), "Cdagdn", i,   "Cdn",    i+1
        os += -1.0/(2*delta^2), "Cdagdn", i+1, "Cdn",    i
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

function run_dmrg(H, psi0; max_bond::Int=500, cutoff::Float64=1e-12, nsweeps::Int=20,
                   use_gpu::Bool=false)
    ramp = [50, 100, 200, 400, 600, 800, 1000]
    dims = vcat(ramp, fill(max_bond, max(nsweeps - length(ramp), 0)))[1:nsweeps]
    dims = min.(dims, max_bond)
    noise = vcat([1e-5, 1e-6, 1e-7, 1e-8], fill(0.0, max(nsweeps-4, 0)))[1:nsweeps]
    sweeps = Sweeps(nsweeps)
    setmaxdim!(sweeps, dims...)
    setcutoff!(sweeps, fill(cutoff, nsweeps)...)
    setnoise!(sweeps, noise...)

    if use_gpu
        @info "Moving H and psi to GPU..."
        H_run = cu(H)
        psi_run = cu(psi0)
    else
        H_run = H
        psi_run = psi0
    end

    energy, psi_out = dmrg(H_run, psi_run, sweeps; outputlevel=1)
    psi_cpu = use_gpu ? NDTensors.cpu(psi_out) : psi_out
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
    run_list = Set(strip.(split(get(ENV, "DMRG_SYSTEMS", "H,He"), ",")))
    max_bond = parse(Int, get(ENV, "DMRG_MAXBOND", "500"))
    nsweeps  = parse(Int, get(ENV, "DMRG_NSWEEPS", "20"))
    use_gpu  = parse(Bool, get(ENV, "USE_GPU", "false"))

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
            H, sites = build_hamiltonian(L, delta, rc, z, r; N=N)
            @printf "MPO bond dim = %d\n" maxlinkdim(H)
            flush(stdout)

            psi0 = make_initial_mps(sites, L, Nup, Ndn)
            energy, psi = run_dmrg(H, psi0; max_bond=max_bond, cutoff=1e-12,
                                    nsweeps=nsweeps, use_gpu=use_gpu)
        end

        results[name] = energy
        ref = get(refs, name, NaN)
        err = energy - ref
        @printf "\n%-4s  E = %.13f   ref = %.13f   err = %+.3e\n" name energy ref err
        flush(stdout)

        h5open("data/$(name)_psi.h5", "w") do f
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
