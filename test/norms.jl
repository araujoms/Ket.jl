@testset "Norms              " begin
    @testset "Square operators" begin
        for T in [Float64, Double64, Float128, BigFloat]
            X = Complex{T}.([1.34 5im 2.6; 5.2 4 + 2.1im 0.1; -4.1 0.9 - 2.2im 1im])
            @test isapprox(schatten_norm(X, 1.32), 12.366965628920612)
            @test isapprox(trace_norm(X), 15.34224630614291)
            @test isapprox(schatten_norm(X, 2), 10.22133063744638)
            @test isapprox(schatten_norm(X, Inf), 9.07815413613693)
            @test isapprox(kyfan_norm(X, 2, 1.7), 10.46707268381498)
            @test isapprox(kyfan_norm(X, 3), schatten_norm(X, 2))
            @test isapprox(kyfan_norm(X, 1), schatten_norm(X, Inf))
            @test isapprox(kyfan_norm(X, 3, 1), trace_norm(X))
        end
    end

    @testset "Rectangular operators" begin
        for T in [Float64, Double64, Float128, BigFloat]
            X = Complex{T}.([1.34 5im 2.6; 5.2 4 + 2.1im 0.1])
            @test isapprox(schatten_norm(X, 2.43), 8.68672362793173)
            @test isapprox(trace_norm(X), 11.8442700353432)
            @test isapprox(schatten_norm(X, 2), 9.00086662494229)
            @test isapprox(schatten_norm(X, Inf), 8.25368317983077)
            @test isapprox(kyfan_norm(X, 2, 4.3), schatten_norm(X, 4.3))
            @test isapprox(kyfan_norm(X, 2), schatten_norm(X, 2))
            @test isapprox(kyfan_norm(X, 1), schatten_norm(X, Inf))
            @test isapprox(kyfan_norm(X, 2, 1), trace_norm(X))
        end
    end
end
