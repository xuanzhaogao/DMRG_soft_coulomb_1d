using ITensors
using ITensorMPS
using CUDA
using Printf

# Small-scale GPU test for He
# rc=3, δ=0.5 → L=12 sites (tiny, just to verify GPU path works)

soft_coulomb(u) = 1.0 / sqrt(1.0 + u^2)

function spread(n::Int, lo::Int, hi::Int)
    n == 0 && return Int[]
    n == 1 && return [div(lo + hi, 2)]
    round.(Int, range(lo, hi, length=n))
end

function build_and_run(; use_gpu::Bool, use_qn::Bool)
    rc = 3.0
    delta = 0.5
    z = 2
    N = 2
    Nup, Ndn = 1, 1

    L = round(Int, 2*rc / delta)
    r = [-rc + (i-1)*delta for i in 1:L]

    tag = use_gpu ? "GPU" : "CPU"
    tag *= use_qn ? "+QN" : "+dense"
    @printf "\n=== He test (%s)  rc=%.1f  δ=%.1f  L=%d ===\n" tag rc delta L
    flush(stdout)

    sites = siteinds("Electron", L; conserve_qns=use_qn)
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

    H = MPO(os, sites)
    @printf "MPO bond dim = %d\n" maxlinkdim(H)

    # Initial state
    state = fill("Emp", L)
    state[div(L, 3)] = "Up"
    state[2 * div(L, 3)] = "Dn"
    psi0 = MPS(sites, state)

    nsweeps = 8
    max_bond = 50
    dims  = fill(max_bond, nsweeps)
    noise = vcat([1e-5, 1e-6, 1e-7], fill(0.0, nsweeps - 3))
    sweeps = Sweeps(nsweeps)
    setmaxdim!(sweeps, dims...)
    setcutoff!(sweeps, fill(1e-10, nsweeps)...)
    setnoise!(sweeps, noise...)

    if use_gpu
        @info "Moving H and psi0 to GPU..."
        H_run   = cu(H)
        psi_run = cu(psi0)
    else
        H_run   = H
        psi_run = psi0
    end

    @printf "Running DMRG (%s)...\n" tag
    flush(stdout)
    @time energy, psi_out = dmrg(H_run, psi_run, sweeps; outputlevel=1)

    @printf "\n%s result: E = %.10f   maxlinkdim = %d\n" tag energy maxlinkdim(psi_out)
    flush(stdout)
    return energy
end

# Run CPU first (with QN), then GPU tests
E_cpu = build_and_run(use_gpu=false, use_qn=true)

# GPU + dense (most likely to work)
E_gpu_dense = build_and_run(use_gpu=true, use_qn=false)

# GPU + QN (experimental, may fail)
try
    E_gpu_qn = build_and_run(use_gpu=true, use_qn=true)
catch e
    @warn "GPU+QN failed (expected — block-sparse on GPU is experimental)" exception=e
end

@printf "\n=== Summary ===\n"
@printf "CPU+QN:       E = %.10f\n" E_cpu
@printf "GPU+dense:    E = %.10f\n" E_gpu_dense
@printf "Difference:   %.2e\n" abs(E_cpu - E_gpu_dense)
