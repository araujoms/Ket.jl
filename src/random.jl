"""
    random_state(d::Integer, k::Integer=d)

Produces a uniformly distributed random quantum state in dimension `d` with rank `k`.

Reference: Życzkowski and Sommers, https://arxiv.org/abs/quant-ph/0012101
"""
function random_state(d::Integer, k::Integer = d; T::Type = Float64, R::Type = Complex{T})
    x = randn(R, (d, k))
    y = x * x'
    return LA.Hermitian(y / LA.tr(y))
end
export random_state

random_state_pure(d::Integer; T::Type = Float64, R::Type = Complex{T}) = random_state(d, 1; T, R)
export random_state_pure

function random_state_pure_vector(d::Integer; T::Type = Float64, R::Type = Complex{T})
    psi = randn(R, d)
    return psi / LA.norm(psi)
end
export random_state_pure_vector

"""
    random_unitary(d::Integer; T::Type, R::Type = Complex{T})

Produces a Haar-random unitary matrix in dimension `d`. If `R` is a real type the output is instead
a Haar-random (real) orthogonal matrix.

Reference: Mezzadri, https://arxiv.org/abs/math-ph/0609050
"""
function random_unitary(d::Integer; T::Type = Float64, R::Type = Complex{T})
    z = randn(R, (d, d))
    fact = LA.qr(z)
    Λ = sign.(real(LA.Diagonal(fact.R)))
    return fact.Q * Λ
end
export random_unitary

"""
    random_povm(d::Integer, n::Integer, r::Integer)

Produces a random POVM of dimension `d` with `n` outcomes and rank `r`.
"""
function random_povm(d::Integer, n::Integer, r::Integer=1; T::Type = Float64, R::Type = Complex{T})
    d <= r * n || throw(ArgumentError("We need d ≤ n*r, but got d = $(d) and n*r = $(n*r)"))
    U = random_unitary(r * n; T, R)
    V = U[:, 1:d]
    E = LA.Hermitian.([V' * kron(LA.I(r), proj(i, n; T, R)) * V for i = 1:n])
    return E
end
export random_povm
