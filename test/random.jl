@testset "Random             " begin
    @testset "States" begin
        ρ = random_state(3)
        @test isa(ρ, Hermitian{ComplexF64})
        ψ = random_state_ket(3)
        @test isa(ψ, Vector{ComplexF64})
        for R in [Float64, Double64, Float128, BigFloat]
            ψ = random_state_ket(R, 3)
            @test ψ' * ψ ≈ 1
            @test isa(ψ, Vector{R})
            ρ = random_state(R, 3)
            @test tr(ρ) ≈ 1
            @test rank(ρ; rtol = Ket._rtol(R)) == 3
            ρ = random_state(R, 3, 2)
            @test tr(ρ) ≈ 1
            @test rank(ρ; rtol = Ket._rtol(R)) == 2
            ρ = random_state(R, 3, 1)
            @test tr(ρ) ≈ 1
            @test rank(ρ; rtol = Ket._rtol(R)) == 1
            @test minimum(eigvals(ρ)) > -Ket._rtol(R)
            @test isa(ρ, Hermitian{R})
            T = Complex{R}
            ψ = random_state_ket(T, 3)
            @test ψ' * ψ ≈ 1
            @test isa(ψ, Vector{T})
            ρ = random_state(R, 3)
            @test tr(ρ) ≈ 1
            @test rank(ρ; rtol = Ket._rtol(R)) == 3
            ρ = random_state(T, 3, 2)
            @test tr(ρ) ≈ 1
            @test rank(ρ; rtol = Ket._rtol(R)) == 2
            ρ = random_state(T, 3, 1)
            @test tr(ρ) ≈ 1
            @test rank(ρ; rtol = Ket._rtol(R)) == 1
            @test minimum(eigvals(ρ)) > -Ket._rtol(R)
            @test isa(ρ, Hermitian{T})
        end
    end
    @testset "Unitaries" begin
        U = random_unitary(3)
        @test isa(U, Matrix{ComplexF64})
        for R in [Float64, Double64, Float128, BigFloat]
            U = random_unitary(R, 3)
            @test U * U' ≈ I(3)
            @test isa(U, Matrix{R})
            T = Complex{R}
            U = random_unitary(T, 3)
            @test U * U' ≈ I(3)
            @test isa(U, Matrix{T})
        end
    end
    @testset "Probability" begin
        p = random_probability(3)
        @test isa(p, Vector{Float64})
        for T in [Float64, Double64, Float128, BigFloat]
            p = random_probability(T, 5)
            @test sum(p) ≈ 1
            @test minimum(p) ≥ 0
            @test isa(p, Vector{T})
        end
    end
    @testset "POVM" begin
        for R in [Float64, Double64, Float128, BigFloat]
            E = random_povm(R, 2, 3)
            @test test_povm(E)
            for i = 1:length(E)
                @test rank(E[i]; rtol = Ket._rtol(R)) == 2
            end
            E = random_povm(R, 2, 3, 1)
            @test test_povm(E)
            for i = 1:length(E)
                @test rank(E[i]; rtol = Ket._rtol(R)) == 1
            end
            T = Complex{R}
            E = random_povm(T, 2, 3)
            @test test_povm(E)
            for i = 1:length(E)
                @test rank(E[i]; rtol = Ket._rtol(R)) == 2
            end
            E = random_povm(T, 2, 3, 1)
            @test test_povm(E)
            for i = 1:length(E)
                @test rank(E[i]; rtol = Ket._rtol(R)) == 1
            end
        end
    end
end
