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
end
