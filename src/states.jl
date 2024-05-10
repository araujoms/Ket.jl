"Produces the maximally entangled state Φ⁺ of local dimension `d`"
function phiplus(::Type{T}, d::Integer = 2) where {T<:Number}
    psi = zeros(T, d^2)
    for i = 1:d
        psi += kron(ket(T, i, d), ket(T, i, d))
    end
    return LA.Hermitian(psi * psi' / T(d))
end
phiplus(d::Integer = 2) = phiplus(ComplexF64, d)
export phiplus # SD: maybe call it ϕ_plus or even ϕp? There is also φ potentially.

"Produces the maximally entangled state ψ⁻ of local dimension `d`"
function psiminus(::Type{T}, d::Integer = 2) where {T<:Number}
    psi = zeros(T, d^2)
    for i = 1:d
        psi += (-1)^(i + 1) * kron(ket(T, i, d), ket(T, d - i + 1, d))
    end
    return LA.Hermitian(psi * psi' / T(d))
end
psiminus(d::Integer = 2) = psiminus(ComplexF64, d)
export psiminus # SD: maybe call it ψ_minus or even ψm?

"Produces the isotropic state of local dimension `d` with visibility `v`"
function isotropic(::Type{T}, v::Real = 1.0, d::Integer = 2) where {T<:Number}
    return LA.Hermitian(v * phiplus(T, d) + (1 - v) * LA.I(d^2) / T(d^2))
end
isotropic(v::Real = 1.0, d::Integer = 2) = isotropic(ComplexF64, v, d)
export isotropic

"Produces the anti-isotropic state of local dimension `d` with visibility `v`"
function anti_isotropic(::Type{T}, v::Real = 1.0, d::Integer = 2) where {T<:Number}
    return LA.Hermitian(v * psiminus(T, d) + (1 - v) * LA.I(d^2) / T(d^2))
end
anti_isotropic(v::Real = 1.0, d::Integer = 2) = anti_isotropic(ComplexF64, v, d)
export anti_isotropic
