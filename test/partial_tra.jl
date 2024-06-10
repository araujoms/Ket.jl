@testset "Partial trace" begin
    d1, d2, d3 = 2, 3, 4
    for R in [Float64, Double64, Float128, BigFloat]
        for T in [R, Complex{R}]
            a = randn(T, d1, d1)
            b = randn(T, d2, d2)
            c = randn(T, d3, d3)
            ab = kron(a, b)
            ac = kron(a, c)
            bc = kron(b, c)
            abc = kron(ab, c)
            @test partial_trace(ab, [1, 2], [d1, d2])[1] ≈ tr(ab)
            @test partial_trace(ab, 2, [d1, d2]) ≈ a * tr(b)
            @test partial_trace(ab, 1, [d1, d2]) ≈ b * tr(a)
            @test partial_trace(ab, Int64[], [d1, d2]) ≈ ab
            @test partial_trace(abc, [1, 2, 3], [d1, d2, d3])[1] ≈ tr(abc)
            @test partial_trace(abc, [2, 3], [d1, d2, d3]) ≈ a * tr(b) * tr(c)
            @test partial_trace(abc, [1, 3], [d1, d2, d3]) ≈ b * tr(a) * tr(c)
            @test partial_trace(abc, [1, 2], [d1, d2, d3]) ≈ c * tr(a) * tr(b)
            @test partial_trace(abc, 3, [d1, d2, d3]) ≈ ab * tr(c)
            @test partial_trace(abc, 2, [d1, d2, d3]) ≈ ac * tr(b)
            @test partial_trace(abc, 1, [d1, d2, d3]) ≈ bc * tr(a)
            @test partial_trace(abc, Int64[], [d1, d2, d3]) ≈ abc
        end
    end
end

@testset "Partial Transpose" begin
    d1, d2, d3 = 2, 3, 4
    for R in [Float64, Double64, Float128, BigFloat]
        for T in [R, Complex{R}]
            a = randn(T, d1, d1)
            b = randn(T, d2, d2)
            c = randn(T, d3, d3)
            ab = kron(a, b)
            ac = kron(a, c)
            bc = kron(b, c)
            abc = kron(ab, c)
            @test partial_transpose(ab, [1, 2], [d1, d2]) ≈ transpose(ab)
            @test partial_transpose(ab, 2, [d1, d2]) ≈ kron(a, transpose(b))
            @test partial_transpose(ab, 1, [d1, d2]) ≈ kron(transpose(a), b)
            @test partial_transpose(ab, Int64[], [d1, d2]) ≈ ab
            @test partial_transpose(abc, [1, 2, 3], [d1, d2, d3]) ≈ transpose(abc)
            @test partial_transpose(abc, [2, 3], [d1, d2, d3]) ≈ kron(a, transpose(b), transpose(c))
            @test partial_transpose(abc, [1, 3], [d1, d2, d3]) ≈ kron(transpose(a), b, transpose(c))
            @test partial_transpose(abc, [1, 2], [d1, d2, d3]) ≈ kron(transpose(a), transpose(b), c)
            @test partial_transpose(abc, 3, [d1, d2, d3]) ≈ kron(ab, transpose(c))
            @test partial_transpose(abc, 2, [d1, d2, d3]) ≈ kron(a, transpose(b), c)
            @test partial_transpose(abc, 1, [d1, d2, d3]) ≈ kron(transpose(a), bc)
            @test partial_transpose(abc, Int64[], [d1, d2, d3]) ≈ abc
        end
    end
end

#TODO add test with JuMP variables
