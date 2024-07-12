@testset "Supermaps          " begin
    for R in [Float64, Double64] #Float128 and BigFloat take too long
        din, dout = 2, 3
        K = [randn(R, dout, din) for _ = 1:3]
        @test diamond_norm(K) ≈ diamond_norm(choi(K), [din, dout]) atol = 1e-8 rtol = sqrt(Base.rtoldefault(R))
        T = Complex{R}
        K = [randn(T, dout, din) for _ = 1:3]
        @test diamond_norm(K) ≈ diamond_norm(choi(K), [din, dout]) atol = 1e-8 rtol = sqrt(Base.rtoldefault(R))
    end
end
