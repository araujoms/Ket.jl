@testset "Entanglement       " begin
    for R in [Float64, Double64, Float128, BigFloat]
        T = Complex{R}
        ψ = random_state_ket(T, 6)
        λ, U, V = schmidt_decomposition(ψ, [2, 3])
        @test vec(Diagonal(λ)) ≈ kron(U', V') * ψ
        ψ = random_state_ket(T, 4)
        λ, U, V = schmidt_decomposition(ψ)
        @test vec(Diagonal(λ)) ≈ kron(U', V') * ψ
    end
end
