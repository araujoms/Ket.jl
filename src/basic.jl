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

const Measurement{T} = Vector{LA.Hermitian{T,Matrix{T}}}
export Measurement

"""
    povm(B::Vector{<:AbstractMatrix{T}})

Creates a set of (projective) measurements from a set of bases given as unitary matrices.
"""
function povm(B::Vector{<:AbstractMatrix{T}}) where {T<:Number}
    return [[ketbra(B[x][:, a]) for a in 1:size(B[x], 2)] for x in eachindex(B)]
end
export povm

"""
    povm(A::Array{T, 4}, n::Vector{Int64})

Converts a set of measurements in the common tensor format into a matrix of matrices.
The second argument is fixed by the size of `A` but can also contain custom number of outcomes.
"""
function povm(A::Array{T,4}, n::Vector{Int64} = fill(size(A, 3), size(A, 4))) where {T<:Number}
    return [[LA.Hermitian(A[:, :, a, x]) for a in 1:n[x]] for x in 1:size(A, 4)]
end

"""
    test_povm(A::Matrix{<:AbstractMatrix{T}})

Checks if the measurement defined by A is valid (hermitian, semi-definite positive, and normalized).
"""
# SD: maybe check_povm instead, but then check_mub and check_sic also I'd say
function test_povm(A::Matrix{<:AbstractMatrix{T}}) where {T<:Number}
    # TODO
end

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
    gell_mann([T=ComplexF64,], d::Integer = 3)

Constructs the set `G` of generalized Gell-Mann matrices in dimension `d` such that
`G[1] = I` and `G[i]*G[j] = 2 δ_ij`.

Reference: [Generalizations of Pauli matrices](https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices)
"""
function gell_mann(::Type{T}, d::Integer = 3) where {T<:Number}
    return [gell_mann(T, i, j, d) for j in 1:d, i in 1:d][:]
    # d=2, ρ = 1/2(σ0 + n*σ)
    # d=3, ρ = 1/3(I + √3 n*λ)
    #      ρ = 1/d(I + sqrt(2/(d*(d-1))) n*λ)
    # SD: the next line would be for a potential KetSparse extension
    # SD: I Haven't thought yet how to deal with this.
    # return [gell_mann(T, k, j, d, sparse(zeros(Complex{T}, d, d))) for j in 1:d, k in 1:d][:]
end
gell_mann(d::Integer = 3) = gell_mann(ComplexF64, d)
export gell_mann

"""
    gell_mann([T=ComplexF64,], i::Integer, j::Integer, d::Integer = 3)

Constructs the set `i`,`j`th Gell-Mann matrix of dimension `d`.

Reference: [Generalizations of Pauli matrices](https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices)
"""
function gell_mann(::Type{T}, i::Integer, j::Integer, d::Integer = 3) where {T<:Number}
    return gell_mann!(zeros(T, d, d), i, j, d)
end
gell_mann(i::Integer, j::Integer, d::Integer = 3) = gell_mann(ComplexF64, i, j, d)

"""
    gell_mann!(res::AbstractMatrix{T}, i::Integer, j::Integer, d::Integer = 3)

In-place version of `gell_mann`.
"""
function gell_mann!(res::AbstractMatrix{T}, i::Integer, j::Integer, d::Integer = 3) where {T<:Number}
    if i < j
        res[i, j] = 1
        res[j, i] = 1
    elseif i > j
        res[i, j] = im
        res[j, i] = -im
    elseif i == 1
        for k in 1:d
            res[k, k] = 1 # _sqrt(T, 2) / _sqrt(T, d) if we want a proper normalisation
        end
    elseif i == d
        tmp = _sqrt(T, 2) / _sqrt(T, d * (d - 1))
        for k in 1:d-1
            res[k, k] = tmp
        end
        res[d, d] = -(d - 1) * tmp
    else
        gell_mann!(view(res, 1:d-1, 1:d-1), i, j, d - 1)
    end
    return res
end
export gell_mann!

_rtol(::Type{T}) where {T<:Number} = Base.rtoldefault(real(T))

_eps(::Type{T}) where {T<:Number} = _realeps(real(T))
_realeps(::Type{T}) where {T<:AbstractFloat} = eps(T)
_realeps(::Type{<:Real}) = 0

"""
    cleanup!(M::AbstractArray{T}; tol = Base.rtoldefault(real(T)))

Zeroes out real or imaginary parts of `M` that are smaller than `tol`.
"""
function cleanup!(M::AbstractArray{T}; tol = _eps(T)) where {T<:Number}
    wrapper = Base.typename(typeof(M)).wrapper
    cleanup!(parent(M); tol)
    return wrapper(M)
end

function cleanup!(M::Array{T}; tol = _eps(T)) where {T<:Number}
    if isbitstype(T)
        M2 = reinterpret(real(T), M) #this is a no-op when T<:Real
        _cleanup!(M2; tol)
    else
        reM = real(M)
        imM = imag(M)
        _cleanup!(reM; tol)
        _cleanup!(imM; tol)
        M .= Complex.(reM, imM)
    end
    return M
end
export cleanup!

function _cleanup!(M; tol)
    return M[abs.(M).<tol] .= 0
end

function applykraus(K, M)
    return sum(LA.Hermitian(Ki * M * Ki') for Ki in K)
end
export applykraus
