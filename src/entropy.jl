_log(base::Real, x) = x > 0 ? log(base, x) : zero(x)

"""
    relative_entropy([base=2,] ρ::AbstractMatrix, σ::AbstractMatrix)

Computes the (quantum) relative entropy tr(`ρ` (log `ρ` - log `σ`)) between positive semidefinite matrices `ρ` and `σ` using a base `base` logarithm. Note that the support of `ρ` must be contained in the support of `σ` but for efficiency this is not checked.

Reference: [Quantum relative entropy](https://en.wikipedia.org/wiki/Quantum_relative_entropy).
"""
function relative_entropy(base::Real, ρ::AbstractMatrix{T}, σ::AbstractMatrix{S}) where {T,S}
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
    logρ_λ = _log.(Ref(base), ρ_λ)
    logσ_λ = _log.(Ref(base), σ_λ)
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
    relative_entropy([base=2,] p::AbstractVector, q::AbstractVector)

Computes the relative entropy D(`p`||`q`) = Σᵢpᵢlog(pᵢ/qᵢ) between two non-negative vectors `p` and `q` using a base `base` logarithm. Note that the support of `p` must be contained in the support of `q` but for efficiency this is not checked.

Reference: [Relative entropy](https://en.wikipedia.org/wiki/Relative_entropy).
"""
function relative_entropy(base::Real, p::AbstractVector{T}, q::AbstractVector{S}) where {T,S<:Real}
    R = promote_type(T, S)
    if length(p) != length(q)
        throw(ArgumentError("`p` and q must have the same length."))
    end
    if any(p .< -Base.rtoldefault(R)) || any(q .< -Base.rtoldefault(R))
        throw(ArgumentError("p and q must be non-negative."))
    end
    logp = _log.(Ref(base), p)
    logq = _log.(Ref(base), q)
    h = sum(p[i] * (logp[i] - logq[i]) for i = 1:length(p))
    return h
end
relative_entropy(p::AbstractVector, q::AbstractVector) = relative_entropy(2, p, q)

"""
    binary_relative_entropy([base=2,] p::Real, q::Real)

Computes the binary relative entropy D(`p`||`q`) = p log(p/q) + (1-p) log((1-p)/(1-q)) between two probabilities `p` and `q` using a base `base` logarithm.

Reference: [Relative entropy](https://en.wikipedia.org/wiki/Relative_entropy).
"""
binary_relative_entropy(base::Real, p::Real, q::Real) = relative_entropy(base, [p, 1 - p], [q, 1 - q])
binary_relative_entropy(p::Real, q::Real) = binary_relative_entropy(2, p, q)
export binary_relative_entropy

"""
    entropy([base=2,] ρ::AbstractMatrix)

Computes the von Neumann entropy -tr(ρ log ρ) of a positive semidefinite operator `ρ` using a base `base` logarithm.

Reference: [von Neumann entropy](https://en.wikipedia.org/wiki/Von_Neumann_entropy).
"""
function entropy(base::Real, ρ::AbstractMatrix)
    if size(ρ, 1) != size(ρ, 2)
        throw(ArgumentError("ρ must be square."))
    end
    λ = LA.eigvals(ρ)
    if any(λ .< -Base.rtoldefault(eltype(λ)))
        throw(ArgumentError("ρ must be positive semidefinite."))
    end
    h = -sum(λ[i] * _log(base, λ[i]) for i = 1:size(ρ, 1))
    return h
end
entropy(ρ::AbstractMatrix) = entropy(2, ρ)
export entropy

"""
    entropy([base=2,] p::AbstractVector)

Computes the Shannon entropy -Σᵢpᵢlog(pᵢ) of a non-negative vector `p` using a base `base` logarithm.

Reference: [Entropy (information theory)](https://en.wikipedia.org/wiki/Entropy_(information_theory)).
"""
function entropy(base::Real, p::AbstractVector{T}) where {T<:Real}
    if any(p .< -Base.rtoldefault(T))
        throw(ArgumentError("p must be non-negative."))
    end
    h = -sum(p[i] * _log(base, p[i]) for i = 1:length(p))
    return h
end
entropy(p::AbstractVector) = entropy(2, p)
export entropy

"""
    binary_entropy([base=2,] p::Real)

Computes the Shannon entropy -p log(p) - (1-p)log(1-p) of a probability `p` using a base `base` logarithm.

Reference: [Entropy (information theory)](https://en.wikipedia.org/wiki/Entropy_(information_theory)).
"""
binary_entropy(base::Real, p::Real) = p == 0 || p == 1 ? zero(p) : -p * log(base, p) - (1 - p) * log(base, 1 - p)
binary_entropy(p::Real) = binary_entropy(2, p)
export binary_entropy

"""
    conditional_entropy([base=2,] pAB::AbstractMatrix)

Computes the conditional (Shannon) entropy H(A|B) of the joint probability distribution `pAB` using a base `base` logarithm.

Reference: [Conditional entropy](https://en.wikipedia.org/wiki/Conditional_entropy).
"""
function conditional_entropy(base::Real, pAB::AbstractMatrix{T}) where {T<:Real}
    nA, nB = size(pAB)
    h = T(0)
    pB = sum(pAB; dims = 1)
    for a = 1:nA, b = 1:nB
        h -= pAB[a, b] * _log(base, pAB[a, b] / pB[b])
    end
    return h
end
conditional_entropy(pAB::AbstractMatrix) = conditional_entropy(2, pAB)
export conditional_entropy
