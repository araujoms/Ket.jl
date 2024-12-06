@testset "States             " begin
    for R in [Float64, Double64, Float128, BigFloat]
        T = Complex{R}
        ψ = state_phiplus_ket(T)
        @test ψ == inv(sqrt(R(2))) * (ket(1, 4) + ket(4, 4))
        @test ketbra(ψ) ≈ state_phiplus(T)
        ψ = state_psiminus_ket(T)
        @test ψ == inv(sqrt(R(2))) * (ket(2, 4) - ket(3, 4))
        @test ketbra(ψ) ≈ state_psiminus(T)
        ψ = state_w_ket(T)
        @test ψ == inv(sqrt(R(3))) * (ket(2, 8) + ket(3, 8) + ket(5, 8))
        @test ketbra(ψ) ≈ state_w(T)
        coeff = T.([1, 2, 2]) / 9
        @test state_w_ket(T; coeff) == T.(ket(2, 8) + 2 * ket(3, 8) + 2 * ket(5, 8)) / 9
        ψ = state_ghz_ket(T)
        @test ψ == inv(sqrt(R(2))) * (ket(1, 8) + ket(8, 8))
        @test ketbra(ψ) ≈ state_ghz(T)
        coeff = [T(3) / 5, T(4) / 5]
        @test state_ghz_ket(T; coeff) == (T(3) * ket(1, 8) + T(4) * ket(8, 8)) / 5
        @test state_super_singlet(T, 2) ≈ state_psiminus(T)
        U = foldl(kron, fill(Matrix(random_unitary(T, 3)), 3))
        rho = state_super_singlet(T, 3)
        @test U * rho * U' ≈ rho
    end
end
