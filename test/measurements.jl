@testset "Measurements" begin
    @testset "SIC POVMs" begin
        for T in [Float16, Float32, Float64, Double64, Float128, BigFloat], d in 1:7
            @test test_sic(sic_povm(d; T))
        end
    end
    @testset "MUBs" begin
        for T in [Float16, Float32, Float64, Double64, Float128, BigFloat]
            @test test_mub(mub(2; T))
            @test test_mub(mub(3; T))
            @test test_mub(mub(4; T))
            @test test_mub(mub(6; T))
            @test test_mub(mub(9; T))
        end
        for T in [Int8, Int16, Int32, Int64, Int128, BigInt]
            @test test_mub(Rational{T}.(mub(4, 2; R = Cyc{Rational{T}})))
            @test test_mub(Complex{Rational{T}}.(mub(4; R = Cyc{Rational{T}})))
        end
        @test test_mub(mub(5, 5, 7; R = Cyc{Rational{BigInt}})) # can access beyond the number of combinations
    end
end
