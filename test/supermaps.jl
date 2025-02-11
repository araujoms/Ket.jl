@testset "Supermaps          " begin
    for R ∈ (Float64, Double64), T ∈ (R, Complex{R}) #Float128 and BigFloat take too long
        γ = R(8)/10
        K = [[1 0;0 √γ], [0 √(1-γ);0 0]]
        ρ = random_state(T, 2)
        damped_ρ = Hermitian([ρ[1,1]+ρ[2,2]*(1-γ) ρ[1,2]*√γ; ρ[2,1]*√γ ρ[2,2]*γ])
        @test applykraus(K, ρ) ≈ damped_ρ
        p = R(7)/10
        @test channel_bit_flip(ρ, p) ≈ Hermitian([ρ[1,1]*p+ρ[2,2]*(1-p) ρ[1,2]*p+ρ[2,1]*(1-p); ρ[2,1]*p+ρ[1,2]*(1-p) ρ[1,1]*(1-p)+ρ[2,2]*p])
        @test channel_phase_flip(ρ, p) ≈ Hermitian([ρ[1,1] ρ[1,2]*(2p-1); ρ[2,1]*(2p-1) ρ[2,2]])
        @test channel_bit_phase_flip(ρ, p) ≈ Hermitian([ρ[1,1]*p+ρ[2,2]*(1-p) ρ[1,2]*p-ρ[2,1]*(1-p); ρ[2,1]*p-ρ[1,2]*(1-p) ρ[1,1]*(1-p)+ρ[2,2]*p])
        @test channel_depolarizing(ρ, p) ≈ Hermitian(ρ*(1-p)+I/2*p)
        @test channel_amplitude_damping(ρ, γ) ≈ Hermitian([ρ[1,1]+ρ[2,2]*γ ρ[1,2]*sqrt(1-γ); ρ[2,1]*sqrt(1-γ) ρ[2,2]*(1-γ)])
        ρ_st = ρ
        for _ in 1:50
            ρ_st = channel_generalized_amplitude_damping(ρ_st, p, γ)
        end
        @test ρ_st ≈ Hermitian([p 0; 0 1-p])
        @test channel_phase_damping(ρ, γ) ≈ channel_phase_flip(ρ, (1+sqrt(1 − γ))/2)
        din, dout = 2, 3
        K = [randn(T, dout, din) for _ ∈ 1:3]
        @test diamond_norm(K) ≈ diamond_norm(choi(K), [din, dout]) atol = 1.0e-8 rtol = sqrt(_rtol(T))
    end
end
