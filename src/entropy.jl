_goodlog(base::Real, x::Real) = x > 0 ? log(base, x) : zero(x)

function relative_entropy(ρ::AbstractMatrix, σ::AbstractMatrix; base::Real = 2)
    T = real(eltype(ρ))
    d = size(ρ, 1)
    ρ_fact = LA.eigen(ρ)
    σ_fact = LA.eigen(σ)
    m = abs2.(ρ_fact.vectors' * σ_fact.vectors)
    logρ = _goodlog.(Ref(base), ρ_fact.values)
    logσ = log.(Ref(base), σ_fact.values)
    h = T(0)
    for j = 1:d
        for i = 1:d
            @views h += ρ_fact.values[i] * (logρ[i] - logσ[j]) * m[i, j]
        end
    end
    return h
end
export relative_entropy

function vonneumann_entropy(ρ::AbstractMatrix; base::Real = 2)
    λ = LA.eigvals(ρ)
    h = -LA.dot(λ, _goodlog.(Ref(base), λ))
    return h
end
export vonneumann_entropy

function conditional_entropy(p::AbstractMatrix{T}; base::Real = 2) where {T<:Real}
    nA, nB = size(p)
    h = T(0)
    pB = sum(p; dims = 2)
    for a = 1:nA
        for b = 1:nB
            h -= p[a, b] * _goodlog(base, p[a, b] / pB[b])
        end
    end
    return h
end
export conditional_entropy

binary_entropy(p; base::Real = 2) = p == 0 || p == 1 ? zero(p) : -p * log(base, p) - (1 - p) * log(base, 1 - p)
export binary_entropy
