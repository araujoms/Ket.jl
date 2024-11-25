@testset "Nonlocal games     " begin
    @test eltype(chsh()) <: Float64
    @test eltype(cglmp()) <: Float64
    @test eltype(inn22()) <: Int
    @test local_bound(chsh()) ≈ 0.75
    @test local_bound(chsh(Int64, 3)) == 6
    @test local_bound(cglmp(Int64, 4)) == 9
    Random.seed!(1337)
    @test seesaw(fp2cg(cglmp()),[3,3,2,2],3)[1] ≈ (15+sqrt(33))/24
    @test seesaw(inn22(),[2,2,3,3],2)[1] ≈ 1.25
    for T in [Float64, Double64, Float128, BigFloat]
        @test eltype(chsh(T)) <: T
        @test eltype(cglmp(T)) <: T
        @test cglmp(T, 4)[3] == T(1) / 12
    end
end

@testset "FP and FC notations" begin
    Aax = povm(mub(2))
    @test correlation_tensor(state_phiplus(), Aax, 2) ≈ Diagonal([1, 1, -1, 1])
    @test correlation_tensor(state_phiplus(), Aax, 2; marg = false) ≈ Diagonal([1, 1, -1])
    scenario = [2,3,4,5]
    cg = randn(scenario[3]*(scenario[1]-1) + 1, scenario[4]*(scenario[2]-1) + 1)
    @test fp2cg(cg2fp(cg,scenario)) ≈ cg
    @test fp2cg(cg2fp(cg,scenario,true),true) ≈ cg
end
