@testset "Incompatibility    " begin
    A = povm(mub(2))
    @test abs(incompatibility_robustness_depolarizing(A) - 1/√3) < 1e-6
    @test abs(incompatibility_robustness_random(A) - 1/√3) < 1e-6
    @test abs(incompatibility_robustness_probabilistic(A) - 1/√3) < 1e-6
    @test abs(incompatibility_robustness_jointly_measurable(A) - (√3 - 1)) < 1e-6
    @test abs(incompatibility_robustness_generalized(A) - (1 + 1/√3) / 2) < 1e-6
    # other types
    @test abs(incompatibility_robustness_depolarizing(povm(broadcast.(Float64, mub(2, 2)))) - 1/√2) < 1e-6
    @test abs(incompatibility_robustness_depolarizing(povm((mub(Complex{Double64}, 2)))) - 1/√3) < 1e-10
end
