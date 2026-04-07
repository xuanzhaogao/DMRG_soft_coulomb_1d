using ITensors
using ITensorMPS
using Printf

soft_coulomb(u) = 1.0 / sqrt(1.0 + u^2)

function spread(n::Int, lo::Int, hi::Int)
    n == 0 && return Int[]
    n == 1 && return [div(lo + hi, 2)]
    round.(Int, range(lo, hi, length=n))
end

function profile_he(; rc=13.0, delta=0.1)
    z, N, Nup, Ndn = 2, 2, 1, 1
    L = round(Int, 2*rc / delta)
    r = [-rc + (i-1)*delta for i in 1:L]

    @printf "He: rc=%.1f  δ=%.2f  L=%d\n" rc delta L
    flush(stdout)

    # Step 1: Build OpSum
    @printf "\n[1] Building OpSum... "
    flush(stdout)
    t1 = @elapsed begin
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
    n_terms = length(os)
    @printf "%.2f s  (%d terms)\n" t1 n_terms
    flush(stdout)

    # Step 2: MPO from OpSum
    @printf "[2] Building MPO from OpSum... "
    flush(stdout)
    t2 = @elapsed begin
        H = MPO(os, sites)
    end
    @printf "%.2f s  (MPO bond dim = %d)\n" t2 maxlinkdim(H)
    flush(stdout)

    # Step 3: Initial MPS
    @printf "[3] Building initial MPS... "
    flush(stdout)
    t3 = @elapsed begin
        state = fill("Emp", L)
        up_pos = spread(Nup, 1, L)
        for i in up_pos; state[i] = "Up"; end
        candidates = setdiff(1:L, up_pos)
        dn_pos = isempty(candidates) ? spread(Ndn, 1, L) :
                 candidates[spread(Ndn, 1, length(candidates))]
        for i in dn_pos; state[i] = (state[i] == "Up") ? "UpDn" : "Dn"; end
        psi0 = MPS(sites, state)
    end
    @printf "%.2f s\n" t3
    flush(stdout)

    # Step 4: DMRG (just 3 sweeps to get timing)
    @printf "[4] Running DMRG (3 sweeps, χ_max=100)... "
    flush(stdout)
    t4 = @elapsed begin
        sweeps = Sweeps(3)
        setmaxdim!(sweeps, 50, 100, 100)
        setcutoff!(sweeps, 1e-10, 1e-10, 1e-10)
        setnoise!(sweeps, 1e-5, 1e-6, 0.0)
        energy, psi = dmrg(H, psi0, sweeps; outputlevel=1)
    end
    @printf "%.2f s\n" t4
    flush(stdout)

    @printf "\n=== Timing Summary ===\n"
    total = t1 + t2 + t3 + t4
    @printf "  OpSum construction:  %8.2f s  (%5.1f%%)\n" t1 (100*t1/total)
    @printf "  MPO construction:    %8.2f s  (%5.1f%%)\n" t2 (100*t2/total)
    @printf "  Initial MPS:         %8.2f s  (%5.1f%%)\n" t3 (100*t3/total)
    @printf "  DMRG (3 sweeps):     %8.2f s  (%5.1f%%)\n" t4 (100*t4/total)
    @printf "  Total:               %8.2f s\n" total
end

profile_he()
