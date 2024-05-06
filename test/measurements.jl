@testset "Measurements" begin
    @testset "SIC POVMs" begin
        for T in [Float64, Double64, Float128, BigFloat], d in 1:7
            @test test_sic(sic_povm(Complex{T}, d))
        end
    end
    @testset "MUBs" begin
        for R in [Float64, Double64, Float128, BigFloat]
            T = Complex{R}
            @test test_mub(mub(T, 2))
            @test test_mub(mub(T, 3))
            @test test_mub(mub(T, 4))
            @test test_mub(mub(T, 6))
            @test test_mub(mub(T, 9))
        end
        for T in [Int64, Int128, BigInt]
            @test test_mub(Rational{T}.(mub(4, 2; R = Cyc{Rational{T}})))
            @test test_mub(Complex{Rational{T}}.(mub(4; R = Cyc{Rational{T}})))
        end
        @test test_mub(mub(5, 5, 7; R = Cyc{Rational{BigInt}})) # can access beyond the number of combinations
    end
end
