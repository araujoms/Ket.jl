"""
    random_state(d::Integer, k::Integer=d)

Produces a uniformly distributed random quantum state in dimension `d` with rank `k`.

Reference: Życzkowski and Sommers, [arXiv:quant-ph/0012101](https://arxiv.org/abs/quant-ph/0012101).
"""
function random_state(::Type{T}, d::Integer, k::Integer = d) where {T}
    x = randn(T, (d, k))
    y = x * x'
    y ./= LA.tr(y)
    return LA.Hermitian(y)
end
random_state(d::Integer, k::Integer = d) = random_state(ComplexF64, d, k)
export random_state

"""
    random_state_vector(d::Integer)

Produces a Haar-random quantum state vector in dimension `d`.

Reference: Życzkowski and Sommers, [arXiv:quant-ph/0012101](https://arxiv.org/abs/quant-ph/0012101).
"""
function random_state_vector(::Type{T}, d::Integer) where {T}
    psi = randn(T, d)
    LA.normalize!(psi)
    return psi
end
random_state_vector(d::Integer) = random_state_vector(ComplexF64, d)
export random_state_vector

"""
    random_unitary(d::Integer; T::Type, R::Type = Complex{T})

Produces a Haar-random unitary matrix in dimension `d`.
If `R` is a real type the output is instead a Haar-random (real) orthogonal matrix.

Reference: Mezzadri, [arXiv:math-ph/0609050](https://arxiv.org/abs/math-ph/0609050).
"""
function random_unitary(::Type{T}, d::Integer) where {T}
    z = randn(T, (d, d))
    Q, R = LA.qr(z)
    Λ = sign.(real(LA.Diagonal(R)))
    return Q * Λ
end
random_unitary(d::Integer) = random_unitary(ComplexF64, d)
export random_unitary

"""
    random_povm(d::Integer, n::Integer, r::Integer)

Produces a random POVM of dimension `d` with `n` outcomes and rank `min(k, d)`.

Reference: Heinosaari et al., [arXiv:1902.04751](https://arxiv.org/abs/1902.04751).
"""
function random_povm(d::Integer, n::Integer, k::Integer = d; T::Type = Float64, R::Type = Complex{T})
    d ≤ n * k || throw(ArgumentError("We need d ≤ n*k, but got d = $(d) and n*k = $(n*k)"))
    E = [Matrix{R}(undef, (d, d)) for _ = 1:n]
    for i = 1:n
        G = randn(R, (d, k))
        LA.mul!(E[i], G, G')
    end
    S = sum(LA.Hermitian.(E))
    rootinvS = sqrt(inv(S)) #don't worry, the probability of getting a singular S is zero
    mat = Matrix{R}(undef, (d, d))
    for i = 1:n
        LA.mul!(mat, rootinvS, E[i])
        LA.mul!(E[i], mat, rootinvS)
    end
    return LA.Hermitian.(E)
end
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
