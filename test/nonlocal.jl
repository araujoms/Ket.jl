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
    for T in [Float64, Double64, Float128, BigFloat]
        Aax = povm(mub(Complex{T}, 2))
        fc_phiplus = Diagonal([1, 1, 1, -1])
        @test tensor_correlation(state_phiplus(Complex{T}), Aax, 2) ≈ fc_phiplus
        @test tensor_correlation(state_phiplus(Complex{T}), Aax, 2; marg = false) ≈ fc_phiplus[2:end, 2:end]
        fc_ghz = zeros(Int64, 4, 4, 4)
        fc_ghz[[1, 6, 18, 21, 43, 48, 60, 63]] .= [1, 1, 1, 1, 1, -1, -1, -1]
        @test tensor_correlation(state_ghz(Complex{T}), Aax, 3) ≈ fc_ghz
        @test tensor_correlation(state_ghz(Complex{T}), Aax, 3; marg = false) ≈ fc_ghz[2:end, 2:end, 2:end]
        m = [3, 4, 5] # dichotomic outcomes
        fc1 = randn(T, m...)
        fc2 = randn(T, m...)
        fp1 = tensor_probability(fc1, true)
        fp2 = tensor_probability(fc2, false)
        @test tensor_correlation(fp1, true) ≈ fc1
        @test tensor_correlation(fp2, false) ≈ fc2
        @test dot(fc1, fc2) ≈ dot(fp1, fp2)
    end
end

@testset "FP and CG notations" begin
    for T in [Float64, Double64, Float128, BigFloat]
        Aax = povm(mub(Complex{T}, 2))
        cg_phiplus = [1.0 0.5 0.5 0.5; 0.5 0.5 0.25 0.25; 0.5 0.25 0.5 0.25; 0.5 0.25 0.25 0.0]
        @test tensor_collinsgisin(state_phiplus(Complex{T}, ), Aax, 2) ≈ cg_phiplus
        scenario = [2, 3, 4, 5]
        cg1 = randn(T, scenario[3] * (scenario[1] - 1) + 1, scenario[4] * (scenario[2] - 1) + 1)
        cg2 = randn(T, scenario[3] * (scenario[1] - 1) + 1, scenario[4] * (scenario[2] - 1) + 1)
        fp1 = tensor_probability(cg1, scenario, true)
        fp2 = tensor_probability(cg2, scenario, false)
        @test tensor_collinsgisin(fp1, true) ≈ cg1
        @test tensor_collinsgisin(fp2, false) ≈ cg2
        @test dot(fp1, fp2) ≈ dot(cg1, cg2)
    end
end
