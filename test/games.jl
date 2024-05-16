@testset "FP and FC notations" begin
    Aax = povm(mub(2))
    @test correlation_tensor(isotropic(), Aax, Aax) ≈ Diagonal([1, 1, -1, 1])
    @test correlation_tensor(isotropic(), Aax, Aax; marg = false) ≈ Diagonal([1, 1, -1])
end
