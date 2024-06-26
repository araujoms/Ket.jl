@testset "Nonlocal" begin
    @test isa(chsh()[1], Float64)
    @test isa(cglmp(3)[1], Float64)
    @test local_bound(chsh()) ≈ 0.75
    @test local_bound(chsh(Int64, 3)) == 6
    @test local_bound(cglmp(Int64, 4)) == 9
    for T in [Float64, Double64, Float128, BigFloat]
        @test isa(chsh(T)[1], T)
        @test isa(cglmp(T, 3)[1], T)
        @test cglmp(T, 4)[3] == T(1) / 12
    end
end
@testset "FP and FC notations" begin
    Aax = povm(mub(2))
    @test correlation_tensor(isotropic(1.0), Aax, Aax) ≈ Diagonal([1, 1, -1, 1])
    @test correlation_tensor(isotropic(1.0), Aax, Aax; marg = false) ≈ Diagonal([1, 1, -1])
end
