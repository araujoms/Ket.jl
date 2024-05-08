_log(b::Real, x::Real) = x > 0 ? log(b, x) : zero(x)

"""
    relative_entropy([b=2,] ρ::AbstractMatrix, σ::AbstractMatrix)

Computes the (quantum) relative entropy tr(`ρ` (log `ρ` - log `σ`)) between positive semidefinite matrices `ρ` and `σ` using a base `b` logarithm. Note that the support of `ρ` must be contained in the support of `σ` but for efficiency this is not checked.

Reference: [Quantum relative entropy](https://en.wikipedia.org/wiki/Quantum_relative_entropy).
"""
function relative_entropy(b::Real, ρ::AbstractMatrix{T}, σ::AbstractMatrix{S}) where {T,S<:Number}
    R = real(promote_type(T, S))
    if size(ρ) != size(σ)
        throw(ArgumentError("ρ and σ have the same size."))
    end
    if size(ρ, 1) != size(ρ, 2)
        throw(ArgumentError("ρ and σ must be square."))
    end
    ρ_λ, ρ_U = LA.eigen(ρ)
    σ_λ, σ_U = LA.eigen(σ)
    if any(ρ_λ .< -Base.rtoldefault(R)) || any(σ_λ .< -Base.rtoldefault(R))
        throw(ArgumentError("ρ and σ must be positive semidefinite."))
    end
    m = abs2.(ρ_U' * σ_U)
    logρ_λ = _log.(Ref(b), ρ_λ)
    logσ_λ = _log.(Ref(b), σ_λ)
    d = size(ρ, 1)
    h = R(0)
    @inbounds for j = 1:d, i = 1:d
        h += ρ_λ[i] * (logρ_λ[i] - logσ_λ[j]) * m[i, j]
    end
    return h
end
relative_entropy(ρ::AbstractMatrix, σ::AbstractMatrix) = relative_entropy(2, ρ, σ)
export relative_entropy

"""
    relative_entropy([b=2,] p::AbstractVector, q::AbstractVector)

Computes the relative entropy D(`p`||`q`) = Σᵢpᵢlog(pᵢ/qᵢ) between two non-negative vectors `p` and `q` using a base `b` logarithm. Note that the support of `p` must be contained in the support of `q` but for efficiency this is not checked.

Reference: [Relative entropy](https://en.wikipedia.org/wiki/Relative_entropy).
"""
function relative_entropy(b::Real, p::AbstractVector{T}, q::AbstractVector{S}) where {T,S<:Real}
    R = promote_type(T, S)
    if length(p) != length(q)
        throw(ArgumentError("`p` and q must have the same length."))
    end
    if any(p .< -Base.rtoldefault(R)) || any(q .< -Base.rtoldefault(R))
        throw(ArgumentError("p and q must be non-negative."))
    end
    logp = _log.(Ref(b), p)
    logq = _log.(Ref(b), q)
    h = sum(p[i] * (logp[i] - logq[i]) for i = 1:length(p))
    return h
end
relative_entropy(p::AbstractVector, q::AbstractVector) = relative_entropy(2, p, q)

"""
    binary_relative_entropy([b=2,] p::Real, q::Real)

Computes the binary relative entropy D(`p`||`q`) = p log(p/q) + (1-p) log((1-p)/(1-q)) between two probabilities `p` and `q` using a base `b` logarithm.

Reference: [Relative entropy](https://en.wikipedia.org/wiki/Relative_entropy).
"""
binary_relative_entropy(b::Real, p::Real, q::Real) = relative_entropy(b, [p, 1 - p], [q, 1 - q])
binary_relative_entropy(p::Real, q::Real) = binary_relative_entropy(2, p, q)
export binary_relative_entropy

"""
    entropy([b=2,] ρ::AbstractMatrix)

Computes the von Neumann entropy -tr(ρ log ρ) of a positive semidefinite operator `ρ` using a base `b` logarithm.

Reference: [von Neumann entropy](https://en.wikipedia.org/wiki/Von_Neumann_entropy).
"""
function entropy(b::Real, ρ::AbstractMatrix)
    if size(ρ, 1) != size(ρ, 2)
        throw(ArgumentError("ρ must be square."))
    end
    λ = LA.eigvals(ρ)
    if any(λ .< -Base.rtoldefault(eltype(λ)))
        throw(ArgumentError("ρ must be positive semidefinite."))
    end
    h = -sum(λ[i] * _log(b, λ[i]) for i = 1:size(ρ, 1))
    return h
end
entropy(ρ::AbstractMatrix) = entropy(2, ρ)
export entropy

"""
    entropy([b=2,] p::AbstractVector)

Computes the Shannon entropy -Σᵢpᵢlog(pᵢ) of a non-negative vector `p` using a base `b` logarithm.

Reference: [Entropy (information theory)](https://en.wikipedia.org/wiki/Entropy_(information_theory)).
"""
function entropy(b::Real, p::AbstractVector)
    if any(p .< -Base.rtoldefault(eltype(p)))
        throw(ArgumentError("p must be non-negative."))
    end
    h = -sum(p[i] * _log(b, p[i]) for i = 1:length(p))
    return h
end
entropy(p::AbstractVector) = entropy(2, p)
export entropy

"""
    binary_entropy([b=2,] p::Real)

Computes the Shannon entropy -p log(p) - (1-p)log(1-p) of a probability `p` using a base `b` logarithm.

Reference: [Entropy (information theory)](https://en.wikipedia.org/wiki/Entropy_(information_theory)).
"""
binary_entropy(b::Real, p::Real) = p == 0 || p == 1 ? zero(p) : -p * log(b, p) - (1 - p) * log(b, 1 - p)
binary_entropy(p::Real) = binary_entropy(2, p)
export binary_entropy

function conditional_entropy(b::Real, p::AbstractMatrix{T}) where {T<:Real}
    nA, nB = size(p)
    h = T(0)
    pB = sum(p; dims = 2)
    for a = 1:nA, b = 1:nB
        h -= p[a, b] * _log(b, p[a, b] / pB[b])
    end
    return h
end
conditional_entropy(b::Real, p::AbstractMatrix) = conditional_entropy(2, p)
export conditional_entropy
