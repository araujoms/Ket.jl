@testset "Incompatibility    " begin
    A = povm(mub(2))
    @test incompatibility_robustness(A; noise = "depolarizing") ≈ 1 / √3
    @test incompatibility_robustness(A; noise = "random") ≈ 1 / √3
    @test incompatibility_robustness(A; noise = "probabilistic") ≈ 1 / √3 rtol = 1e-7
    @test incompatibility_robustness(A; noise = "jointly_measurable") ≈ (√3 - 1)
    @test incompatibility_robustness(A; noise = "general") ≈ (1 + 1 / √3) / 2
    # other types, also checks that the default noise is "general"
    @test incompatibility_robustness(povm(broadcast.(Float64, mub(2, 2)))) ≈ (1 + 1 / √2) / 2
    @test incompatibility_robustness(povm((mub(Complex{Double64}, 2)))) ≈ (1 + 1 / √Double64(3)) / 2
end
