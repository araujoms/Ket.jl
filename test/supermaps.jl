@testset "Supermaps          " begin
    for R ∈ (Float64, Double64), T ∈ (R, Complex{R}) #Float128 and BigFloat take too long
        γ = R(8)/10
        K = [[1 0;0 √γ], [0 √(1-γ);0 0]]
        ρ = random_state(T, 2)
        damped_ρ = Hermitian([ρ[1,1]+ρ[2,2]*(1-γ) ρ[1,2]*√γ; ρ[2,1]*√γ ρ[2,2]*γ])
        @test applykraus(K, ρ) ≈ damped_ρ
        din, dout = 2, 3
        K = [randn(T, dout, din) for _ ∈ 1:3]
        @test diamond_norm(K) ≈ diamond_norm(choi(K), [din, dout]) atol = 1.0e-8 rtol = sqrt(_rtol(T))
    end
end
