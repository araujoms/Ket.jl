@testset "Supermaps          " begin
    for R ∈ (Float64, Double64), T ∈ (R, Complex{R}) #Float128 and BigFloat take too long
        din, dout = 2, 3
        K = [randn(T, dout, din) for _ ∈ 1:3]
        @test diamond_norm(K) ≈ diamond_norm(choi(K), [din, dout]) atol = 1.0e-8 rtol = sqrt(_rtol(T))
    end
end
