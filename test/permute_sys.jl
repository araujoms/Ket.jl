@testset "Permute systems" begin

    @testset "Vectors" begin
        d1, d2, d3 = 2, 3, 4
        for R in [Float64, Double64, Float128, BigFloat]
            for T in [R, Complex{R}]
                u = randn(T, d1)
                v = randn(T, d2)
                w = randn(T, d3)
                uv = kron(u, v)
                uw = kron(u, w)
                vw = kron(v, w)
                uvw = kron(u, v, w)
                @test permute_systems(uv, [1, 2], [d1, d2]) ≈ kron(u, v)
                @test permute_systems(uv, [2, 1], [d1, d2]) ≈ kron(v, u)
                @test permute_systems(uw, [2, 1], [d1, d3]) ≈ kron(w, u)
                @test permute_systems(vw, [2, 1], [d2, d3]) ≈ kron(w, v)
                @test permute_systems(uvw, [1,2,3], [d1, d2, d3]) ≈ kron(u, v, w) 
                @test permute_systems(uvw, [2,3,1], [d1, d2, d3]) ≈ kron(v, w, u)
                @test permute_systems(uvw, [3,1,2], [d1, d2, d3]) ≈ kron(w, u, v)
                @test permute_systems(uvw, [1,3,2], [d1, d2, d3]) ≈ kron(u, w, v)
                @test permute_systems(uvw, [2,1,3], [d1, d2, d3]) ≈ kron(v, u, w)
                @test permute_systems(uvw, [3,2,1], [d1, d2, d3]) ≈ kron(w, v, u)
            end
        end
    end

    @testset "Square matrices" begin
        d1, d2, d3 = 2, 3, 4
        for R in [Float64, Double64, Float128, BigFloat]
            for T in [R, Complex{R}]
                a = randn(T, d1, d1)
                b = randn(T, d2, d2)
                c = randn(T, d3, d3)
                ab = kron(a, b)
                ac = kron(a, c)
                bc = kron(b, c)
                abc = kron(a, b, c)
                @test permute_systems(ab, [1, 2], [d1, d2]) ≈ kron(a, b)
                @test permute_systems(ab, [2, 1], [d1, d2]) ≈ kron(b, a)
                @test permute_systems(ac, [2, 1], [d1, d3]) ≈ kron(c, a)
                @test permute_systems(bc, [2, 1], [d2, d3]) ≈ kron(c, b)
                @test permute_systems(abc, [1,2,3], [d1, d2, d3]) ≈ kron(a, b, c)
                @test permute_systems(abc, [2,3,1], [d1, d2, d3]) ≈ kron(b, c, a)
                @test permute_systems(abc, [3,1,2], [d1, d2, d3]) ≈ kron(c, a, b)
                @test permute_systems(abc, [1,3,2], [d1, d2, d3]) ≈ kron(a, c, b)
                @test permute_systems(abc, [2,1,3], [d1, d2, d3]) ≈ kron(b, a, c)
                @test permute_systems(abc, [3,2,1], [d1, d2, d3]) ≈ kron(c, b, a)
            end
        end
    end    

    @testset "Rectangular matrices" begin
        d1, d2, d3 = 2, 3, 4
        for R in [Float64, Double64, Float128, BigFloat]
            for T in [R, Complex{R}]
                a = randn(T, d1, d2)
                b = randn(T, d1, d3)
                c = randn(T, d2, d3)
                ab = kron(a, b)
                ac = kron(a, c)
                bc = kron(b, c)
                abc = kron(a, b, c)
                @test permute_systems(ab, [1, 2], [d1 d2; d1 d3]) ≈ kron(a, b)
                @test permute_systems(ab, [2, 1], [d1 d2; d1 d3]) ≈ kron(b, a)
                @test permute_systems(ac, [2, 1], [d1 d2; d2 d3]) ≈ kron(c, a)
                @test permute_systems(bc, [2, 1], [d1 d3; d2 d3]) ≈ kron(c, b)
                @test permute_systems(abc, [1,2,3], [d1 d2; d1 d3; d2 d3]) ≈ kron(a, b, c)
                @test permute_systems(abc, [2,3,1], [d1 d2; d1 d3; d2 d3]) ≈ kron(b, c, a)
                @test permute_systems(abc, [3,1,2], [d1 d2; d1 d3; d2 d3]) ≈ kron(c, a, b)
                @test permute_systems(abc, [1,3,2], [d1 d2; d1 d3; d2 d3]) ≈ kron(a, c, b)
                @test permute_systems(abc, [2,1,3], [d1 d2; d1 d3; d2 d3]) ≈ kron(b, a, c)
                @test permute_systems(abc, [3,2,1], [d1 d2; d1 d3; d2 d3]) ≈ kron(c, b, a)
            end
        end
                
    end

    
end