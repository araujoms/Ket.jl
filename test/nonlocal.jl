@testset "Nonlocal games     " begin
    @test eltype(chsh()) <: Float64
    @test eltype(cglmp()) <: Float64
    @test eltype(inn22()) <: Int
    @test local_bound(chsh()) ≈ 0.75
    @test local_bound(chsh(Int64, 3)) == 6
    @test local_bound(cglmp(Int64, 4)) == 9
    Random.seed!(1337)
    @test seesaw(tensor_collinsgisin(cglmp()), [3, 3, 2, 2], 3)[1] ≈ (15 + sqrt(33)) / 24
    @test seesaw(inn22(), [2, 2, 3, 3], 2)[1] ≈ 1.25
    for T in [Float64, Double64, Float128, BigFloat]
        @test eltype(chsh(T)) <: T
        @test eltype(cglmp(T)) <: T
        @test cglmp(T, 4)[3] == T(1) / 12
    end
end

@testset "FP and FC notations" begin
    Aax = povm(mub(2))
    @test tensor_correlation(state_phiplus(), Aax, 2) ≈ Diagonal([1, 1, 1, -1])
    @test tensor_correlation(state_phiplus(), Aax, 2; marg = false) ≈ Diagonal([1, 1, -1])
    FC_ghz = zeros(4, 4, 4)
    FC_ghz[[1, 6, 18, 21, 43, 48, 60, 63]] .= [1, 1, 1, 1, -1, 1, 1, 1]
    @test tensor_correlation(state_ghz(), Aax, 3) ≈ FC_ghz
    @test tensor_correlation(state_ghz(), Aax, 3; marg = false) ≈ FC_ghz[2:end, 2:end, 2:end]
    scenario = [3, 4, 5] # dichotomic outcomes to test full correlation
    rho = random_state(2^3)
    mesA = [random_povm(2, 2) for _ in 1:scenario[1]]
    mesB = [random_povm(2, 2) for _ in 1:scenario[2]]
    mesC = [random_povm(2, 2) for _ in 1:scenario[3]]
    FP_behaviour = tensor_probability(rho, mesA, mesB, mesC)
    FP_functional = randn(2, 2, 2, scenario...)
    @test dot(tensor_correlation(FP_behaviour, true), tensor_correlation(FP_functional, false)) ≈ dot(FP_behaviour, FP_functional)
    scenario = [2, 3, 4, 5]
    cg = randn(scenario[3] * (scenario[1] - 1) + 1, scenario[4] * (scenario[2] - 1) + 1)
    @test tensor_collinsgisin(tensor_probability(cg, scenario)) ≈ cg
    @test tensor_collinsgisin(tensor_probability(cg, scenario, true), true) ≈ cg
end
