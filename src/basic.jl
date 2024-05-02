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

"""
    ketbra(v::AbstractVector)

Produces a ketbra of vector `v`.
"""
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

"""
    shift_operator(d::Integer, p::Integer = 1)

Constructs the shift operator X of dimension `d` to the power `p`.
"""
function shift_operator(d::Integer, p::Integer = 1; T::Type = Float64, R::Type = Complex{T})
    X = zeros(R, d, d)
    for i in 0:d-1
        X[mod(i + p, d)+1, i+1] = 1
    end
    return X
end
export shift_operator

"""
    clock_operator(d::Integer, p::Integer = 1)

Constructs the clock operator Z of dimension `d` to the power `p`.
"""
function clock_operator(d::Integer, p::Integer = 1; T::Type = Float64)
    z = zeros(Complex{T}, d)
    ω = exp(im * 2 * T(π) / d)
    for i in 0:d-1
        exponent = mod(i * p, d)
        if exponent == 0
            z[i+1] = 1
        elseif 4 * exponent == d
            z[i+1] = im
        elseif 2 * exponent == d
            z[i+1] = -1
        elseif 4 * exponent == 3 * d
            z[i+1] = -im
        else
            z[i+1] = ω^exponent
        end
    end
    return LA.Diagonal(z)
end
export clock_operator

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
