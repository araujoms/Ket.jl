@testset "Random" begin
    @testset "States" begin
        ρ = random_state(3)
        @test isa(ρ, Hermitian{ComplexF64})
        ψ = random_state_vector(3)
        @test isa(ψ, Vector{ComplexF64})
        for T in [Float64, Double64, Float128, BigFloat]
            ψ = random_state_vector(T, 3)
            @test ψ' * ψ ≈ 1
            @test isa(ψ, Vector{T})
            ρ = random_state(T, 3)
            @test tr(ρ) ≈ 1
            @test rank(ρ) == 3
            ρ = random_state(T, 3, 2)
            @test tr(ρ) ≈ 1
            @test rank(ρ) == 2
            ρ = random_state(T, 3, 1)
            @test tr(ρ) ≈ 1
            @test rank(ρ) == 1
            @test minimum(eigvals(ρ)) > -10 * eps(T)
            @test isa(ρ, Hermitian{T})
            R = Complex{T}
            ψ = random_state_vector(R, 3)
            @test ψ' * ψ ≈ 1
            @test isa(ψ, Vector{R})
            ρ = random_state(R, 3)
            @test tr(ρ) ≈ 1
            @test rank(ρ) == 3
            ρ = random_state(R, 3, 2)
            @test tr(ρ) ≈ 1
            @test rank(ρ) == 2
            ρ = random_state(R, 3, 1)
            @test tr(ρ) ≈ 1
            @test rank(ρ) == 1
            @test minimum(eigvals(ρ)) > -10 * eps(T)
            @test isa(ρ, Hermitian{R})
        end
    end
    @testset "Unitaries" begin
        U = random_unitary(3)
        @test isa(U, Matrix{ComplexF64})
        for T in [Float64, Double64, Float128, BigFloat]
            U = random_unitary(T, 3)
            @test U * U' ≈ I(3)
            @test isa(U, Matrix{T})
            R = Complex{T}
            U = random_unitary(R, 3)
            @test U * U' ≈ I(3)
            @test isa(U, Matrix{R})
        end
    end
end
