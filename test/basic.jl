@testset "Basic" begin
    @testset "Kets" begin
        @test isa(ket(1, 3), Vector{ComplexF64})
        @test isa(proj(1, 3), Hermitian{ComplexF64})
        for R in [Int64, Float64, Double64, Float128, BigFloat]
            ψ = ket(R, 2, 3)
            P = proj(R, 2, 3)
            @test ψ == [0, 1, 0]
            @test isa(ψ, Vector{R})
            @test isa(P, Hermitian{R})
            @test P == ketbra(ψ)
            T = Complex{R}
            ψ = ket(T, 2, 3)
            P = proj(T, 2, 3)
            @test ψ == [0, 1, 0]
            @test isa(ψ, Vector{T})
            @test isa(P, Hermitian{T})
            @test P == ketbra(ψ)
        end
    end
    @testset "Shift and clock" begin
        @test isa(shift(4), Matrix{ComplexF64})
        @test clock(4) == Diagonal([1, im, -1, -im])
        @test isa(clock(4), Diagonal{ComplexF64})
        for R in [Float64, Double64, Float128, BigFloat]
            T = Complex{R}
            @test shift(T, 3) == [0 0 1; 1 0 0; 0 1 0]
            @test clock(T, 3) ≈ Diagonal([1, exp(2 * T(π) * im / 3), exp(-2 * T(π) * im / 3)])
            @test shift(T, 3, 2) == shift(T, 3)^2
            @test clock(T, 3, 2) ≈ clock(T, 3)^2
        end
    end
    @testset "Cleanup" begin
        for T in [
            Float64,
            Double64,
            Float128,
            BigFloat,
            ComplexF64,
            Complex{Double64},
            Complex{Float128},
            Complex{BigFloat},
            QuaternionF64
        ]
            a = randn(T, 2, 2)
            b = Hermitian(a)
            c = UpperTriangular(a)
            d = Diagonal(diag(a))
            @test a ≈ cleanup!(a)
            @test b ≈ cleanup!(b)
            @test c ≈ cleanup!(c)
            @test d ≈ cleanup!(d)
        end
    end
end
