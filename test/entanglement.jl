@testset "Entanglement       " begin
    @testset "Schmidt decomposition" begin
        for R ∈ (Float64, Double64, Float128, BigFloat)
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
        for R ∈ (Float64, Double64), T ∈ (R, Complex{R}) #Float128 and BigFloat take too long
            Random.seed!(8) #makes all states entangled
            ψ = random_state_ket(T, 6)
            @test entanglement_entropy(ψ, [2, 3]) ≈ entanglement_entropy(ketbra(ψ), [2, 3])[1] atol = 1.0e-3 rtol =
                1.0e-3
            ρ = random_state(T, 4)
            h, σ = entanglement_entropy(ρ)
            @test Ket._test_entanglement_entropy_qubit(h, ρ, σ)
        end
    end
    @testset "DPS hierarchy" begin
        for R ∈ (Float64, Double64), T ∈ (R, Complex{R})
            ρ = state_ghz(T, 2, 2)
            s, W = random_robustness(ρ)
            @test eltype(W) == T
            @test s ≈ 0.5 atol = 1.0e-5 rtol = 1.0e-5
            @test dot(ρ, W) ≈ -s atol = 1.0e-5 rtol = 1.0e-5
        end
        d = 3
        @test isapprox(schmidt_number(state_ghz(ComplexF64, d, 2), 2), 1 / 15, atol = 1.0e-3, rtol = 1.0e-3)
        @test isapprox(
            schmidt_number(state_ghz(Float64, d, 2), 2, [d, d], 2; solver = SCS.Optimizer),
            1 / 15,
            atol = 1.0e-3,
            rtol = 1.0e-3
        )
        @test schmidt_number(random_state(Float64, 6), 2, [2, 3], 1; solver = SCS.Optimizer) ≤ 0
    end
    @testset "GME entanglement" begin
        for R ∈ (Float64, Double64)
            ρ = state_ghz(R, 2, 3)

            v, W = ppt_mixture(ρ, [2, 2, 2])
            @test isapprox(v, 0.4285, atol = 1.0e-3)
            full_body_basis = collect(Iterators.flatten(n_body_basis(i, 3) for i ∈ 0:3))
            v, w = ppt_mixture(ρ, [2, 2, 2], full_body_basis)
            @test isapprox(v, 0.4285, atol = 1.0e-3)
            @test isapprox(sum(w[i] * full_body_basis[i] for i ∈ eachindex(w)), W, atol = 1.0e-3)

            two_body_basis = collect(Iterators.flatten(n_body_basis(i, 3) for i ∈ 0:2))
            v, w = ppt_mixture(state_w(3), [2, 2, 2], two_body_basis)
            @test isapprox(v, 0.696, atol = 1.0e-3)
        end
    end
end
