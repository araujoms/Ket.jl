@testset "Measurements" begin
    @testset "SIC POVMs" begin
        for T in [Float64, Double64, Float128, BigFloat], d in 1:7
            @test test_sic(sic_povm(Complex{T}, d))
        end
    end
    @testset "MUBs" begin
        for T in [Int8, Int64, BigInt]
            @test test_mub(mub(T(6)))
        end
        for R in [Float64, Double64, Float128, BigFloat]
            T = Complex{R}
            @test test_mub(mub(T, 2))
            @test test_mub(mub(T, 3))
            @test test_mub(mub(T, 4))
            @test test_mub(mub(T, 6))
            @test test_mub(mub(T, 9))
        end
        for T in [Int64, Int128, BigInt]
            @test test_mub(broadcast.(Rational{T}, mub(Cyc{Rational{T}}, 4, 2)))
            @test test_mub(broadcast.(Complex{Rational{T}}, mub(Cyc{Rational{T}}, 4)))
        end
        @test test_mub(mub(Cyc{Rational{BigInt}}, 5, 5, 7)) # can access beyond the number of combinations
    end
end
