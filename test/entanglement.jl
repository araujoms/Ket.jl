@testset "Entanglement       " begin
    @testset "Schmidt decomposition" begin
        for R in [Float64, Double64, Float128, BigFloat]
            T = Complex{R}
            ψ = random_state_ket(T, 6)
            λ, U, V = schmidt_decomposition(ψ, [2, 3])
            @test vec(Diagonal(λ)) ≈ kron(U', V') * ψ
            ψ = random_state_ket(T, 4)
            λ, U, V = schmidt_decomposition(ψ)
            @test vec(Diagonal(λ)) ≈ kron(U', V') * ψ
        end
    end
    @testset "Entanglement entropy" begin
        for R in [Float64, Double64] #Float128 and BigFloat take too long
            Random.seed!(8) #makes all states entangled
            ψ = random_state_ket(R, 6)
            @test entanglement_entropy(ψ, [2, 3]) ≈ entanglement_entropy(ketbra(ψ), [2, 3])[1] atol = 1e-3 rtol = 1e-3
            ρ = random_state(R, 4)
            h, σ = entanglement_entropy(ρ)
            @test Ket._test_entanglement_entropy_qubit(h, ρ, σ)
            T = Complex{R}
            ψ = random_state_ket(T, 6)
            @test entanglement_entropy(ψ, [2, 3]) ≈ entanglement_entropy(ketbra(ψ), [2, 3])[1] atol = 1e-3 rtol = 1e-3
            ρ = random_state(T, 4)
            h, σ = entanglement_entropy(ρ)
            @test Ket._test_entanglement_entropy_qubit(h, ρ, σ)
        end
    end
    @testset "DPS hierarchy" begin
        for R in [Float64, Double64]
            ρ = state_ghz(R, 2, 2)
            s, W = random_robustness(ρ)
            @test eltype(W) == R
            @test s ≈ 0.5 atol = 1e-5 rtol = 1e-5
            @test dot(ρ, W) ≈ -s atol = 1e-5 rtol = 1e-5
            T = Complex{R}
            ρ = state_ghz(T, 2, 2)
            s, W = random_robustness(ρ)
            @test eltype(W) == T
            @test s ≈ 0.5 atol = 1e-5 rtol = 1e-5
            @test dot(ρ, W) ≈ -s atol = 1e-5 rtol = 1e-5
        end
        @test isapprox(
            schmidt_number(state_ghz(ComplexF64, 3, 2), 2),
            0.625,
            atol = 1e-3,
            rtol = 1e-3
        )
        @test isapprox(
            schmidt_number(state_ghz(Float64, 3, 2), 2, [3, 3], 2; solver = SCS.Optimizer),
            0.625,
            atol = 1e-3,
            rtol = 1e-3
        )
        @test isapprox(
            schmidt_number(random_state(Float64, 6), 2, [2, 3], 1; solver = SCS.Optimizer),
            1.0,
            atol = 1e-3,
            rtol = 1e-3
        )
    end
end
