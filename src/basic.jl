"""
    ket([T=ComplexF64,] i::Integer, d::Integer)

Produces a ket of dimension `d` with nonzero element `i`.
"""
function ket(::Type{T}, i::Integer, d::Integer) where {T<:Number}
    psi = zeros(T, d)
    psi[i] = 1
    return psi
end
ket(i::Integer, d::Integer) = ket(ComplexF64, i, d)
export ket

"""
    ketbra(v::AbstractVector)

Produces a ketbra of vector `v`.
"""
function ketbra(v::AbstractVector)
    return LA.Hermitian(v * v')
end
export ketbra

"""
    proj([T=ComplexF64,] i::Integer, d::Integer)

Produces a projector onto the basis state `i` in dimension `d`.
"""
function proj(::Type{T}, i::Integer, d::Integer) where {T<:Number}
    p = LA.Hermitian(zeros(T, d, d))
    p[i, i] = 1
    return p
end
proj(i::Integer, d::Integer) = proj(ComplexF64, i, d)
export proj

"""
    shift([T=ComplexF64,] d::Integer, p::Integer = 1)

Constructs the shift operator X of dimension `d` to the power `p`.

Reference: [Generalized Clifford algebra](https://en.wikipedia.org/wiki/Generalized_Clifford_algebra)
"""
function shift(::Type{T}, d::Integer, p::Integer = 1) where {T<:Number}
    X = zeros(T, d, d)
    for i in 0:d-1
        X[mod(i + p, d)+1, i+1] = 1
    end
    return X
end
shift(d::Integer, p::Integer = 1) = shift(ComplexF64, d, p)
export shift

"""
    clock([T=ComplexF64,] d::Integer, p::Integer = 1)

Constructs the clock operator Z of dimension `d` to the power `p`.

Reference: [Generalized Clifford algebra](https://en.wikipedia.org/wiki/Generalized_Clifford_algebra)
"""
function clock(::Type{T}, d::Integer, p::Integer = 1) where {T<:Number}
    z = zeros(T, d)
    ω = _root_unity(T, d)
    for i in 0:d-1
        exponent = mod(i * p, d)
        if exponent == 0
            z[i+1] = 1
        elseif 4exponent == d
            z[i+1] = im
        elseif 2exponent == d
            z[i+1] = -1
        elseif 4exponent == 3d
            z[i+1] = -im
        else
            z[i+1] = ω^exponent
        end
    end
    return LA.Diagonal(z)
end
clock(d::Integer, p::Integer = 1) = clock(ComplexF64, d, p)
export clock

"""
    GellMann([T=ComplexF64,], d::Integer = 3)

Constructs the set `G` of generalized Gell-Mann matrices in dimension `d` such that
`G[1] = I` and `G[i]*G[j] = 2 δ_ij`.

Reference: [Generalizations of Pauli matrices](https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices)
"""
# d=2, ρ = 1/2(σ0 + n*σ)
# d=3, ρ = 1/3(I + √3 n*λ)
#      ρ = 1/d(I + sqrt(2/(d*(d-1))) n*λ)
function GellMann(::Type{T}, d::Integer = 3) where {T<:Number}
    return [GellMann(T, k, j, d) for j in 1:d, k in 1:d][:]
    # SD: the next line would be for a potential KetSparse extension
    # SD: I Haven't thought yet how to deal with this.
    # return [GellMann(T, k, j, d, sparse(zeros(Complex{T}, d, d))) for j in 1:d, k in 1:d][:]
end
GellMann(d::Integer = 3) = GellMann(ComplexF64, d)
export GellMann

# SD: maybe it could be wise to add a ! here as res gets modified in place
function GellMann(::Type{T}, k::Integer, j::Integer, d::Integer = 3, res::AbstractMatrix{T} = zeros(T, d, d)) where {T<:Number}
    if k < j
        res[k, j] = 1
        res[j, k] = 1
    elseif k > j
        res[k, j] = im
        res[j, k] = -im
    elseif k == 1
        for i in 1:d
            res[i, i] = 1 # _sqrt(T, 2) / _sqrt(T, d) if we want a proper normalisation
        end
    elseif k == d
        tmp = _sqrt(T, 2) / _sqrt(T, d * (d - 1))
        for i in 1:d-1
            res[i, i] = tmp
        end
        res[d, d] = -(d - 1) * tmp
    else
        GellMann(T, k, j, d - 1, view(res, 1:d-1, 1:d-1))
    end
    return res
end
GellMann(k::Integer, j::Integer, d::Integer = 3) = GellMann(ComplexF64, k, j, d)

_tol(::Type{T}) where {T<:Number} = Base.rtoldefault(real(T))

"""
    cleanup!(M::AbstractArray{T}; tol = Base.rtoldefault(real(T)))

Zeroes out real or imaginary parts of `M` that are smaller than `tol`.
"""
function cleanup!(M::AbstractArray{T}; tol = _tol(T)) where {T<:Number} # SD: is it type stable?
    wrapper = Base.typename(typeof(M)).wrapper
    cleanup!(parent(M); tol)
    return wrapper(M)
end

function cleanup!(M::Array{T}; tol = _tol(T)) where {T<:Number}
    M2 = reinterpret(T, M) #this is a no-op when T<:Real
    _cleanup!(M2; tol)
    return M
end
export cleanup!

function _cleanup!(M::Array; tol)
    return M[abs.(M).<tol] .= 0
end

function applykraus(K, M)
    return sum(LA.Hermitian(Ki * M * Ki') for Ki in K)
end
export applykraus
