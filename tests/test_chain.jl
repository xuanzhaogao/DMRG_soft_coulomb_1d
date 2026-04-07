include("sog_coeffs.jl")

function test_chain()
    delta = 0.5

    # Simulate MPO chain product for 3-hop SOG (sites 1→4, distance = 3δ = 1.5)
    total = 0.0
    for k in 1:N_SOG_REAL
        lam = exp(-SOG_REAL_S[k] * delta)
        wt  = SOG_REAL_M[k]
        # start: lam×Ntot, propagate×2: lam×I, end: wt×Ntot
        contrib = lam * lam^2 * wt
        total += contrib
    end
    for k in 1:N_SOG_PAIR
        rk = exp(-SOG_PAIR_ALPHA[k] * delta)
        th = SOG_PAIR_BETA[k] * delta
        pk = SOG_PAIR_P[k]; qk = SOG_PAIR_Q[k]
        ct = cos(th); st = sin(th)

        # Start vector: [rk*ct, rk*st]
        va = rk * ct; vb = rk * st
        # Propagate twice (2×2 rotation)
        for _ in 1:2
            va_new =  rk*ct * va - rk*st * vb
            vb_new =  rk*st * va + rk*ct * vb
            va = va_new; vb = vb_new
        end
        # End: dot with [2*pk, 2*qk]
        contrib = va * 2*pk + vb * 2*qk
        total += contrib
    end

    exact = 1.0 / sqrt(1.0 + 1.5^2)
    println("MPO chain product (3 hops) = $total")
    println("v(1.5) exact = $exact")
    println("Diff = $(total - exact)")

    # Also test 1-hop (sites 2→3, distance = δ = 0.5)
    total1 = 0.0
    for k in 1:N_SOG_REAL
        lam = exp(-SOG_REAL_S[k] * delta)
        wt  = SOG_REAL_M[k]
        total1 += lam * wt  # start × end, no propagation
    end
    for k in 1:N_SOG_PAIR
        rk = exp(-SOG_PAIR_ALPHA[k] * delta)
        th = SOG_PAIR_BETA[k] * delta
        pk = SOG_PAIR_P[k]; qk = SOG_PAIR_Q[k]
        ct = cos(th); st = sin(th)

        va = rk * ct; vb = rk * st
        total1 += va * 2*pk + vb * 2*qk
    end

    exact1 = 1.0 / sqrt(1.0 + 0.5^2)
    println("\nMPO chain product (1 hop) = $total1")
    println("v(0.5) exact = $exact1")
    println("Diff = $(total1 - exact1)")
end

test_chain()
