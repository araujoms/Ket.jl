"Produces the maximally entangled state Φ⁺ of local dimension `d`"
function phiplus(::Type{T}, d::Integer) where {T<:Number}
    psi = zeros(T, d^2)
    for i = 1:d
        psi += kron(ket(T, i, d), ket(T, i, d))
    end
    return LA.Hermitian(psi * psi' / T(d))
end
phiplus(d::Integer) = phiplus(ComplexF64, d)
export phiplus # SD: maybe call it ϕ_plus or even ϕp? There is also φ potentially.

"Produces the maximally entangled state ψ⁻ of local dimension `d`"
function psiminus(::Type{T}, d::Integer) where {T<:Number}
    psi = zeros(T, d^2)
    for i = 1:d
        psi += (-1)^(i + 1) * kron(ket(T, i, d), ket(T, d - i + 1, d))
    end
    return LA.Hermitian(psi * psi' / T(d))
end
phiminus(d::Integer) = phiminus(ComplexF64, d)
export psiminus # SD: maybe call it ψ_minus or even ψm?

"Produces the isotropic state of local dimension `d` with visibility `v`"
function iso(::Type{T}, v::Real, d::Integer) where {T<:Number}
    return LA.Hermitian(v * phiplus(T, d) + (1 - v) * LA.I(d^2) / T(d^2))
end
iso(v::Real, d::Integer) = iso(ComplexF64, v, d)
export iso

"Produces the anti-isotropic state of local dimension `d` with visibility `v`"
function anti_iso(::Type{T}, v::Real, d::Integer) where {T<:Number}
    return LA.Hermitian(v * psiminus(T, d) + (1 - v) * LA.I(d^2) / T(d^2))
end
anti_iso(v::Real, d::Integer) = anti_iso(ComplexF64, v, d)
export anti_iso
