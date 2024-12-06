@testset "Random             " begin
    @testset "States" begin
        ρ = random_state(3)
        @test isa(ρ, Hermitian{ComplexF64})
        ψ = random_state_ket(3)
        @test isa(ψ, Vector{ComplexF64})
        for R in [Float64, Double64, Float128, BigFloat]
            for f in (identity, complex)
                Random.seed!(1337) #make ranks behave
                ψ = random_state_ket(f(R), 3)
                @test ψ' * ψ ≈ 1
                @test isa(ψ, Vector{f(R)})
                ρ = random_state(f(R), 3)
                @test tr(ρ) ≈ 1
                @test rank(ρ; rtol = Ket._rtol(f(R))) == 3
                ρ = random_state(f(R), 3, 2)
                @test tr(ρ) ≈ 1
                @test rank(ρ; rtol = Ket._rtol(f(R))) == 2
                ρ = random_state(f(R), 3, 1)
                @test tr(ρ) ≈ 1
                @test rank(ρ; rtol = Ket._rtol(f(R))) == 1
                @test minimum(eigvals(ρ)) > -Ket._rtol(f(R))
                @test isa(ρ, Hermitian{f(R)})
            end
        end
    end
    @testset "Unitaries" begin
        U = random_unitary(3)
        @test eltype(U) <: ComplexF64
        for R in [Float64, Double64, Float128, BigFloat]
            for f in (identity, complex)
                U = random_unitary(f(R), 3)
                @test U * U' ≈ I(3)
                @test eltype(U) <: f(R)
            end
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
            for f in (identity, complex)
                E = random_povm(f(R), 2, 3)
                @test test_povm(E)
                for i in 1:length(E)
                    @test rank(E[i]; rtol = Ket._rtol(f(R))) == 2
                end
                E = random_povm(f(R), 2, 3, 1)
                @test test_povm(E)
                for i in 1:length(E)
                    @test rank(E[i]; rtol = Ket._rtol(f(R))) == 1
                end
            end
        end
    end
end
