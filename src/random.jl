#often we don't actually need the variance of the normal variables to be 1, so we don't need to waste time diving everything by sqrt(2)
_randn(::Type{Complex{T}}) where {T<:AbstractFloat} = Complex{T}(randn(T), randn(T))
_randn(::Type{T}) where {T} = randn(T)
_randn(::Type{T}, dim1::Integer, dims::Integer...) where {T} = _randn!(Array{T}(undef, dim1, dims...))
function _randn!(A::AbstractArray{T}) where {T}
    for i ∈ eachindex(A)
        @inbounds A[i] = _randn(T)
    end
    return A
end

"""
    random_state([T=ComplexF64,] d::Integer, k::Integer = d)

Produces a uniformly distributed random quantum state in dimension `d` with rank `k`.

Reference: Życzkowski and Sommers, [arXiv:quant-ph/0012101](https://arxiv.org/abs/quant-ph/0012101).
"""
function random_state(::Type{T}, d::Integer, k::Integer = d) where {T}
    x = _randn(T, d, k)
    y = x * x'
    y ./= tr(y)
    return Hermitian(y)
end
random_state(d::Integer, k::Integer = d) = random_state(ComplexF64, d, k)
export random_state

"""
    random_state_ket([T=ComplexF64,] d::Integer)

Produces a Haar-random quantum state vector in dimension `d`.

Reference: Życzkowski and Sommers, [arXiv:quant-ph/0012101](https://arxiv.org/abs/quant-ph/0012101).
"""
function random_state_ket(::Type{T}, d::Integer) where {T}
    psi = _randn(T, d)
    normalize!(psi)
    return psi
end
random_state_ket(d::Integer) = random_state_ket(ComplexF64, d)
export random_state_ket

#dedicated type for using producing random unitaries using Stewart's algorithm
import Base.size
import LinearAlgebra.lmul!
import LinearAlgebra.rmul!
struct StewartQ{T,S<:LinearAlgebra.QRPackedQ{T},C<:Vector{T}} <: LinearAlgebra.AbstractQ{T}
    q::S
    signs::C
end
size(Q::StewartQ, dim::Integer) = size(Q.q, dim)
size(Q::StewartQ) = size(Q.q)

lmul!(A::StewartQ, B::AbstractVecOrMat) = lmul!(A.q, lmul!(LinearAlgebra.Diagonal(A.signs), B))
lmul!(adjA::LinearAlgebra.AdjointQ{<:Any,<:StewartQ}, B::AbstractVecOrMat) =
    lmul!(LinearAlgebra.Diagonal(adjA.Q.signs), lmul!(adjA.Q.q', B))
rmul!(A::AbstractVecOrMat, B::StewartQ) = rmul!(rmul!(A, B.q), LinearAlgebra.Diagonal(B.signs))
rmul!(A::AbstractVecOrMat, adjB::LinearAlgebra.AdjointQ{<:Any,<:StewartQ}) =
    rmul!(rmul!(A, LinearAlgebra.Diagonal(adjB.Q.signs)), adjB.Q.q')

"""
    random_unitary([T=ComplexF64,] d::Integer)

Produces a Haar-random unitary matrix in dimension `d`.
If `T` is a real type the output is instead a Haar-random (real) orthogonal matrix.

Reference: Stewart, [doi:10.1137/0717034](https://doi.org/10.1137/0717034).
"""
function random_unitary(::Type{T}, d::Integer) where {T<:Number}
    z = Matrix{T}(undef, d, d)
    @inbounds for j ∈ 1:d
        for i ∈ j:d
            z[i, j] = _randn(T)
        end
    end
    τ = Vector{T}(undef, d)
    s = Vector{T}(undef, d)
    @inbounds for k ∈ 1:d #this is a partial QR decomposition where we don't apply the reflection to the rest of the matrix
        @views x = z[k:d, k]
        τ[k] = LinearAlgebra.reflector!(x)
        s[k] = sign(real(x[1]))
    end
    return StewartQ(LinearAlgebra.QRPackedQ(z, τ), s)
end
random_unitary(d::Integer) = random_unitary(ComplexF64, d)
export random_unitary

"""
    random_povm([T=ComplexF64,] d::Integer, n::Integer, k::Integer)

Produces a random POVM of dimension `d` with `n` outcomes and rank `min(k, d)`.

Reference: Heinosaari et al., [arXiv:1902.04751](https://arxiv.org/abs/1902.04751).
"""
function random_povm(::Type{T}, d::Integer, n::Integer, k::Integer = d) where {T<:Number}
    d ≤ n * k || throw(ArgumentError("We need d ≤ n*k, but got d = $(d) and n*k = $(n * k)"))
    G = [randn(T, d, k) for _ ∈ 1:n]
    S = zeros(T, d, d)
    for i ∈ 1:n
        mul!(S, G[i], G[i]', true, true)
    end
    rootinvS = Hermitian(S)^-0.5 #don't worry, the probability of getting a singular S is zero
    E = [Matrix{T}(undef, d, d) for _ ∈ 1:n]
    temp = Matrix{T}(undef, d, k)
    for i ∈ 1:n
        mul!(temp, rootinvS, G[i])
        mul!(E[i], temp, temp')
    end
    return Hermitian.(E)
end
random_povm(d::Integer, n::Integer, k::Integer = d) = random_povm(ComplexF64, d, n, k)
export random_povm

"""
    random_probability([T=Float64,] d::Integer)

Produces a random probability vector of dimension `d` uniformly distributed on the simplex.

Reference: [Dirichlet distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution#Random_variate_generation)
"""
function random_probability(::Type{T}, d::Integer) where {T}
    p = rand(T, d)
    p .= log.(p)
    #p .*= -1 not needed
    p ./= sum(p)
    return p
end
random_probability(d::Integer) = random_probability(Float64, d)
export random_probability
