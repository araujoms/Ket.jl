"""
    phiplus([T=ComplexF64,] d::Integer = 2)

Produces the maximally entangled state Φ⁺ of local dimension `d`.
"""
function phiplus(::Type{T}, d::Integer = 2) where {T<:Number}
    psi = zeros(T, d^2)
    for i = 1:d
        psi += kron(ket(T, i, d), ket(T, i, d))
    end
    return LA.Hermitian(psi * psi' / d)
end
phiplus(d::Integer = 2) = phiplus(ComplexF64, d)
export phiplus

"""
    psiminus([T=ComplexF64,] d::Integer = 2)

Produces the maximally entangled state ψ⁻ of local dimension `d`.
"""
function psiminus(::Type{T}, d::Integer = 2) where {T<:Number}
    psi = zeros(T, d^2)
    for i = 1:d
        psi += (-1)^(i + 1) * kron(ket(T, i, d), ket(T, d - i + 1, d))
    end
    return LA.Hermitian(psi * psi' / d)
end
psiminus(d::Integer = 2) = psiminus(ComplexF64, d)
export psiminus

"""
    isotropic(v::Real, d::Integer = 2)

Produces the isotropic state of local dimension `d` with visibility `v`.
"""
function isotropic(v::T, d::Integer = 2) where {T<:Real}
    return LA.Hermitian(v * phiplus(T, d) + (1 - v) * LA.I(d^2) / T(d^2))
end
export isotropic

"""
    anti_isotropic(v::Real, d::Integer = 2)

Produces the anti-isotropic state of local dimension `d` with visibility `v`.
"""
function anti_isotropic(v::T, d::Integer = 2) where {T<:Real}
    return LA.Hermitian(v * psiminus(T, d) + (1 - v) * LA.I(d^2) / T(d^2))
end
export anti_isotropic
