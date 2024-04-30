"""
    ket(i::Integer, d::Integer; T=ComplexF64)

Produces a ket of dimension `d` with nonzero element `i`.
"""
function ket(i::Integer, d::Integer; T::Type = Float64, R::Type = Complex{T})
    psi = zeros(R, d)
    psi[i] = 1
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
    ketbra[i, i] = 1
    return ketbra
end
export proj

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

"Zeroes out real or imaginary parts of M that are smaller than `tol`"
function cleanup!(M::Array{Complex{T}}; tol::T = Base.rtoldefault(T)) where {T<:Real}
    M2 = reinterpret(T, M)
    _cleanup!(M2; tol)
    return M
end
export cleanup!

function cleanup!(M::Array{T}; tol::T = Base.rtoldefault(T)) where {T<:AbstractFloat}
    _cleanup!(M; tol)
    return M
end

function cleanup!(M::AbstractArray{T}; tol::T = T(0)) where {T<:Number}
    return M
end

function cleanup!(
    M::Union{AbstractMatrix{Complex{T}},AbstractMatrix{T}};
    tol::T = Base.rtoldefault(T)
) where {T<:AbstractFloat}
    wrapper = Base.typename(typeof(M)).wrapper
    cleanup!(parent(M); tol)
    return wrapper(M)
end

function _cleanup!(M::AbstractArray{T}; tol::T = Base.rtoldefault(T)) where {T<:AbstractFloat}
    return M[abs.(M).<tol] .= 0
end

function applykraus(K, M)
    return sum(LA.Hermitian(Ki * M * Ki') for Ki in K)
end
export applykraus
