"Produces the maximally entangled state Φ⁺ of local dimension `d`"
function phiplus(d::Integer; T::Type = Float64, R::Type = Complex{T})
    psi = zeros(R, d^2)
    for i = 1:d
        psi += kron(ket(i, d; T), ket(i, d; T))
    end
    return LA.Hermitian(psi * psi' / d)
end
export phiplus

"Produces the maximally entangled state ψ⁻ of local dimension `d`"
function psiminus(d::Integer; T::Type = Float64, R::Type = Complex{T})
    psi = zeros(R, d^2)
    for i = 1:d
        psi += (-1)^(i + 1) * kron(ket(i, d; T), ket(d - i + 1, d; T))
    end
    return LA.Hermitian(psi * psi' / d)
end
export psiminus

"Produces the isotropic state of local dimension `d` with visibility `v`"
function iso(v::Real, d::Integer; T::Type = Float64)
    return LA.Hermitian(v * phiplus(d; T) + (1 - v) * LA.I(d^2) / T(d^2))
end
export iso

"Produces the anti-isotropic state of local dimension `d` with visibility `v`"
function anti_iso(v::Real, d::Integer; T::Type = Float64)
    return LA.Hermitian(v * psiminus(d; T) + (1 - v) * LA.I(d^2) / T(d^2))
end
export anti_iso
