"""
    seesaw(CG::Matrix, scenario::AbstractVecOrTuple, d::Integer)

Maximizes bipartite Bell functional ∈ Collins-Gisin notation `CG` using the seesaw heuristic. `scenario` is a vector detailing the number of inputs and outputs, ∈ the order [oa, ob, ia, ib].
`d` is an integer determining the local dimension of the strategy.

If `oa` == `ob` == 2 the heuristic reduces to a bunch of eigenvalue problems. Otherwise semidefinite programming is needed and we use the assemblage version of seesaw.

References: Pál and Vértesi, [arXiv:1006.3032](https://arxiv.org/abs/1006.3032),
section II.B.1 of Tavakoli et al., [arXiv:2307.02551](https://arxiv.org/abs/2307.02551)
"""
function seesaw(CG::Matrix{T}, scenario::AbstractVecOrTuple{<:Integer}, d::Integer) where {T<:Real}
    R = _solver_type(T)
    CG = R.(CG)
    T2 = Complex{R}
    minimumincrease = _rtol(R)
    maxiter = 100

    oa, ob, ia, ib = scenario

    if oa == 2 && ob == 2
        λ = T2.(sqrt.(random_probability(R, d)))
        B = [random_povm(T2, d, 2)[1] for _ ∈ 1:ib]
        A = [Hermitian(zeros(T2, d, d)) for _ ∈ 1:ia]
        local ψ
        ω = -R(Inf)
        i = 1
        while true
            _optimise_alice_projectors!(CG, λ, A, B)
            _optimise_bob_projectors!(CG, λ, A, B)
            new_ω = _optimise_state!(CG, λ, A, B)
            if new_ω - ω <= minimumincrease || i > maxiter
                ω = new_ω
                ψ = state_phiplus_ket(T2, d; coeff = λ)
                A = [[A[x]] for x ∈ 1:ia] #rather inconvenient format
                B = [[B[y]] for y ∈ 1:ib] #but consistent with the general case
                break
            end
            ω = new_ω
            i += 1
        end
    else
        B = Vector{Measurement{T2}}(undef, ib)
        for y ∈ 1:ib
            B[y] = random_povm(T2, d, ob)[1:ob-1]
        end
        local ψ, A
        ω = -R(Inf)
        i = 1
        while true
            new_ω, ρxa, ρ_B = _optimise_alice_assemblage(CG, scenario, B)
            new_ω, B = _optimise_bob_povm(CG, scenario, ρxa, ρ_B)
            if new_ω - ω <= minimumincrease || i > maxiter
                ω = new_ω
                ψ, A = _decompose_assemblage(scenario, ρxa, ρ_B)
                break
            end
            ω = new_ω
            i += 1
        end
    end
    return ω, ψ, A, B
end
export seesaw

function _optimise_alice_assemblage(CG::Matrix{R}, scenario, B; solver = Hypatia.Optimizer{R}) where {R<:AbstractFloat}
    oa, ob, ia, ib = scenario
    d = size(B[1][1], 1)

    model = JuMP.GenericModel{R}()
    ρxa = [[JuMP.@variable(model, [1:d, 1:d] ∈ JuMP.HermitianPSDCone()) for _ ∈ 1:oa-1] for _ ∈ 1:ia] #assemblage
    ρ_B = JuMP.@variable(model, [1:d, 1:d], Hermitian) #auxiliary quantum state

    JuMP.@constraint(model, tr(ρ_B) == 1)
    for x ∈ 1:ia
        JuMP.@constraint(model, ρ_B - sum(ρxa[x][a] for a ∈ 1:oa-1) ∈ JuMP.HermitianPSDCone())
    end

    ω = _compute_value_assemblage(CG, scenario, ρxa, ρ_B, B)
    JuMP.@objective(model, Max, ω)

    JuMP.set_optimizer(model, solver)
    JuMP.set_silent(model)
    JuMP.optimize!(model)
    value_ρxa = [[Hermitian(JuMP.value.(ρxa[x][a])) for a ∈ 1:oa-1] for x ∈ 1:ia]
    return JuMP.value(ω), value_ρxa, Hermitian(JuMP.value.(ρ_B))
end

function _optimise_bob_povm(CG::Matrix{R}, scenario, ρxa, ρ_B; solver = Hypatia.Optimizer{R}) where {R<:AbstractFloat}
    oa, ob, ia, ib = scenario
    d = size(ρ_B, 1)

    model = JuMP.GenericModel{R}()
    B = [[JuMP.@variable(model, [1:d, 1:d] ∈ JuMP.HermitianPSDCone()) for _ ∈ 1:ob-1] for _ ∈ 1:ib] #povm
    for y ∈ 1:ib
        JuMP.@constraint(model, I - sum(B[y][b] for b ∈ 1:ob-1) ∈ JuMP.HermitianPSDCone())
    end

    ω = _compute_value_assemblage(CG, scenario, ρxa, ρ_B, B)
    JuMP.@objective(model, Max, ω)

    JuMP.set_optimizer(model, solver)
    JuMP.set_silent(model)
    JuMP.optimize!(model)
    B = [[Hermitian(JuMP.value.(B[y][b])) for b ∈ 1:ob-1] for y ∈ 1:ib]
    return JuMP.value(ω), B
end

function _compute_value_assemblage(CG::Matrix{R}, scenario, ρxa, ρ_B, B) where {R<:AbstractFloat}
    oa, ob, ia, ib = scenario
    aind(a, x) = 1 + a + (x - 1) * (oa - 1)
    bind(b, y) = 1 + b + (y - 1) * (ob - 1)

    ω = CG[1, 1] * one(JuMP.GenericAffExpr{R,JuMP.GenericVariableRef{R}})
    for a ∈ 1:oa-1
        for x ∈ 1:ia
            tempB = sum(CG[aind(a, x), bind(b, y)] * B[y][b] for b ∈ 1:ob-1 for y ∈ 1:ib)
            ω += real(dot(tempB, ρxa[x][a]))
        end
    end
    for a ∈ 1:oa-1
        for x ∈ 1:ia
            ω += CG[aind(a, x), 1] * real(tr(ρxa[x][a]))
        end
    end
    for b ∈ 1:ob-1
        for y ∈ 1:ib
            ω += CG[1, bind(b, y)] * real(dot(B[y][b], ρ_B))
        end
    end
    return ω
end

#rather unstable
function _decompose_assemblage(scenario, ρxa, ρ_B::AbstractMatrix{T}) where {T}
    oa, ob, ia, ib = scenario

    d = size(ρ_B, 1)
    λ, U = eigen(ρ_B)
    ψ = zeros(T, d^2)
    for i ∈ 1:d
        @views ψ .+= sqrt(λ[i]) * kron(conj(U[:, i]), U[:, i])
    end
    invrootλ = map(x -> x >= _rtol(T) ? 1 / sqrt(x) : zero(x), λ)
    W = U * Diagonal(invrootλ) * U'
    A = [[Hermitian(conj(W * ρxa[x][a] * W)) for a ∈ 1:oa-1] for x ∈ 1:ia]
    return ψ, A
end

function _optimise_alice_projectors!(CG::Matrix, λ::Vector, A, B)
    ia, ib = size(CG) .- 1
    d = length(λ)
    for x ∈ 1:ia
        for j ∈ 1:d
            for i ∈ 1:j
                A[x].data[i, j] = CG[x+1, 1] * (i == j) * abs2(λ[i])
                for y ∈ 1:ib
                    A[x].data[i, j] += CG[x+1, y+1] * λ[i] * conj(λ[j] * B[y][i, j])
                end
            end
        end
        _positive_projection!(A[x])
    end
end

function _optimise_bob_projectors!(CG::Matrix, λ::Vector, A, B)
    ia, ib = size(CG) .- 1
    d = length(λ)
    for y ∈ 1:ib
        for j ∈ 1:d
            for i ∈ 1:j
                B[y].data[i, j] = CG[1, y+1] * (i == j) * abs2(λ[i])
                for x ∈ 1:ia
                    B[y].data[i, j] += CG[x+1, y+1] * λ[i] * conj(λ[j] * A[x][i, j])
                end
            end
        end
        _positive_projection!(B[y])
    end
end

function _positive_projection!(M::AbstractMatrix{T}) where {T}
    λ, U = eigen!(M)
    fill!(M, 0)
    for i ∈ 1:length(λ)
        if λ[i] > _rtol(T)
            @views M.data .+= ketbra(U[:, i])
        end
    end
    return M
end

function _optimise_state!(CG::Matrix, λ::Vector{T}, A, B) where {T}
    ia, ib = size(CG) .- 1
    d = length(λ)
    M = Hermitian(zeros(T, d, d))
    for x ∈ 1:ia
        M += CG[x+1, 1] * real(Diagonal(A[x]))
    end
    for y ∈ 1:ib
        M += CG[1, y+1] * real(Diagonal(B[y]))
    end
    for y ∈ 1:ib
        for x ∈ 1:ia
            for j ∈ 1:d
                for i ∈ 1:j
                    M.data[i, j] += CG[x+1, y+1] * A[x].data[i, j] * B[y].data[i, j]
                end
            end
        end
    end
    vals, U = eigen!(M)
    λ .= U[:, d]
    return CG[1, 1] + vals[d]
    #TODO: for large matrices it's perhaps better to do
    #decomp, history = ArnoldiMethod.partialschur(M, nev = 1)
    #λ = decomp.Q[:,1]
end
