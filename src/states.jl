"""
    ket(i::Integer, d::Integer; T=ComplexF64)
Produces a ket of dimension `d` with nonzero element `i`."""
function ket(i::Integer, d::Integer; T::Type = Float64, R::Type = Complex{T})
    psi = zeros(R, d)
    psi[i] = R(1)
    return psi
end
export ket

function ketbra(v::AbstractVector)
    return LA.Hermitian(v * v')
end
export ketbra

"Produces a projector onto the basis state `i` in dimension `d`."
function proj(i::Integer, d::Integer; T::Type = Float64, R::Type = Complex{T})
    ketbra = LA.Hermitian(zeros(R, d, d))
    ketbra[i, i] = R(1)
    return ketbra
end
export proj

"Produces the maximally entangled state Φ⁺ of local dimension `d`"
function phiplus(d::Integer; T::Type = Float64, R::Type = Complex{T})
    psi = zeros(R, d^2)
    for i = 1:d
        psi += kron(ket(i, d; T), ket(i, d; T))
    end
    return LA.Hermitian(psi * psi' / T(d))
end
export phiplus

"Produces the maximally entangled state ψ⁻ of local dimension `d`"
function psiminus(d::Integer; T::Type = Float64, R::Type = Complex{T})
    psi = zeros(R, d^2)
    for i = 1:d
        psi += (-1)^(i + 1) * kron(ket(i, d; T), ket(d - i + 1, d; T))
    end
    return LA.Hermitian(psi * psi' / T(d))
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

"Zeroes out real or imaginary parts of M that are smaller than `eps`"
function cleanup!(M::Array{R}; eps::Real = 1e-10) where {R<:Complex}
    M2 = reinterpret(real(R), M)
    _cleanup!(M2; eps)
    return M
end
export cleanup!

function cleanup!(M::Array{<:Real}; eps::Real = 1e-10)
    _cleanup!(M; eps)
    return M
end

function cleanup!(M::AbstractMatrix; eps::Real = 1e-10)
    wrapper = Base.typename(typeof(M)).wrapper
    cleanup!(parent(M); eps)
    return wrapper(M)
end

function _cleanup!(M; eps::Real = 1e-10)
    return M[abs.(M).<eps] .= 0
end

function applykraus(K,M)
    return sum(LA.Hermitian(Ki*M*Ki') for Ki in K)
end
export applykraus
