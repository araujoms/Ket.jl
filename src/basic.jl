"""
    ket([T=ComplexF64,] i::Integer, d::Integer = 2)

Produces a ket of dimension `d` with nonzero element `i`.
"""
function ket(::Type{T}, i::Integer, d::Integer = 2) where {T<:Number}
    psi = zeros(T, d)
    psi[i] = 1
    return psi
end
ket(i::Integer, d::Integer = 2) = ket(Bool, i, d)
export ket

"""
    ketbra(v::AbstractVector)

Produces a ketbra of vector `v`.
"""
function ketbra(v::AbstractVector)
    return Hermitian(v * v')
end
export ketbra

"""
    proj([T=ComplexF64,] i::Integer, d::Integer = 2)

Produces a projector onto the basis state `i` in dimension `d`.
"""
function proj(::Type{T}, i::Integer, d::Integer = 2) where {T<:Number}
    p = Hermitian(zeros(T, d, d))
    p[i, i] = 1
    return p
end
proj(i::Integer, d::Integer = 2) = proj(Bool, i, d)
export proj

const Measurement{T} = Vector{Hermitian{T,Matrix{T}}}
export Measurement

"""
    povm(B::Vector{<:AbstractMatrix{T}})

Creates a set of (projective) measurements from a set of bases given as unitary matrices.
"""
function povm(B::Vector{<:AbstractMatrix})
    return [[ketbra(B[x][:, a]) for a ∈ 1:size(B[x], 2)] for x ∈ eachindex(B)]
end
export povm

"""
    tensor_to_povm(A::Array{T,4}, o::Vector{Int})

Converts a set of measurements in the common tensor format into a matrix of (hermitian) matrices.
By default, the second argument is fixed by the size of `A`.
It can also contain custom number of outcomes if there are measurements with less outcomes.
"""
function tensor_to_povm(Aax::Array{T,4}, o::Vector{Int} = fill(size(Aax, 3), size(Aax, 4))) where {T}
    return [[Hermitian(Aax[:, :, a, x]) for a ∈ 1:o[x]] for x ∈ axes(Aax, 4)]
end
export tensor_to_povm

"""
    povm_to_tensor(Axa::Vector{<:Measurement})

Converts a matrix of (hermitian) matrices into a set of measurements in the common tensor format.
"""
function povm_to_tensor(Axa::Vector{Measurement{T}}) where {T<:Number}
    d, o, m = _measurements_parameters(Axa)
    Aax = zeros(T, d, d, maximum(o), m)
    for x ∈ eachindex(Axa)
        for a ∈ eachindex(Axa[x])
            Aax[:, :, a, x] .= Axa[x][a]
        end
    end
    return Aax
end
export povm_to_tensor

function _measurements_parameters(Axa::Vector{Measurement{T}}) where {T<:Number}
    @assert !isempty(Axa)
    # dimension on which the measurements act
    d = size(Axa[1][1], 1)
    # tuple of outcome numbers
    o = Tuple(length.(Axa))
    # number of inputs, i.e., of mesurements
    m = length(Axa)
    return d, o, m
end
_measurements_parameters(Aa::Measurement) = _measurements_parameters([Aa])

"""
    test_povm(A::Vector{<:AbstractMatrix{T}})

Checks if the measurement defined by A is valid (hermitian, semi-definite positive, and normalized).
"""
function test_povm(E::Vector{<:AbstractMatrix{T}}) where {T<:Number}
    !all(ishermitian.(E)) && return false
    d = size(E[1], 1)
    !(sum(E) ≈ I(d)) && return false
    for i ∈ 1:length(E)
        minimum(eigvals(E[i])) < -_rtol(T) && return false
    end
    return true
end
export test_povm

"""
    shift([T=ComplexF64,] d::Integer, p::Integer = 1)

Constructs the shift operator X of dimension `d` to the power `p`.

Reference: [Generalized Clifford algebra](https://en.wikipedia.org/wiki/Generalized_Clifford_algebra)
"""
function shift(::Type{T}, d::Integer, p::Integer = 1) where {T<:Number}
    X = zeros(T, d, d)
    for i ∈ 0:(d-1)
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
    for i ∈ 0:(d-1)
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
    return Diagonal(z)
end
clock(d::Integer, p::Integer = 1) = clock(ComplexF64, d, p)
export clock

"""
    pauli([T=ComplexF64,], ind::Vector{<:Integer})

Constructs the Pauli matrices: 0 or "I" for the identity,
1 or "X" for the Pauli X operation, 2 or "Y" for the Pauli Y
operator, and 3 or "Z" for the Pauli Z operator.
Vectors of integers between 0 and 3 or strings of I, X, Y, Z
automatically generate Kronecker products of the corresponding
operators.
"""
function pauli(::Type{T}, i::Integer) where {T<:Number}
    return gellmann(T, i ÷ 2 + 1, i % 2 + 1, 2)
end
function pauli(::Type{T}, ind::Vector{<:Integer}) where {T<:Number}
    if length(ind) == 1
        return pauli(T, ind[1])
    else
        return kron([pauli(T, i) for i ∈ ind]...)
    end
end
function pauli(::Type{T}, str::String) where {T<:Number}
    ind = Int[]
    for c ∈ str
        if c in ['I', 'i', '1']
            push!(ind, 0)
        elseif c in ['X', 'x']
            push!(ind, 1)
        elseif c in ['Y', 'y']
            push!(ind, 2)
        elseif c in ['Z', 'z']
            push!(ind, 3)
        else
            @warn "Unknown character"
        end
    end
    return pauli(T, ind)
end
pauli(::Type{T}, c::Char) where {T<:Number} = pauli(T, string(c))
pauli(i::Integer) = pauli(ComplexF64, i)
pauli(ind::Vector{<:Integer}) = pauli(ComplexF64, ind)
pauli(str::String) = pauli(ComplexF64, str)
pauli(c::Char) = pauli(ComplexF64, c)
export pauli

"""
    gellmann([T=ComplexF64,], d::Integer = 3)

Constructs the set `G` of generalized Gell-Mann matrices in dimension `d` such that
`G[1] = I` and `Tr(G[i]*G[j]) = 2 δ_ij`.

Reference: [Generalizations of Pauli matrices](https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices)
"""
function gellmann(::Type{T}, d::Integer = 3) where {T<:Number}
    return [gellmann(T, i, j, d) for j ∈ 1:d, i ∈ 1:d][:]
    # d=2, ρ = 1/2(σ0 + n*σ)
    # d=3, ρ = 1/3(I + √3 n*λ)
    #      ρ = 1/d(I + sqrt(2/(d*(d-1))) n*λ)
    # SD: the next line would be for a potential KetSparse extension
    # SD: I Haven't thought yet how to deal with this.
    # return [gellmann(T, k, j, d, sparse(zeros(Complex{T}, d, d))) for j in 1:d, k in 1:d][:]
end
gellmann(d::Integer = 3) = gellmann(ComplexF64, d)
export gellmann

"""
    gellmann([T=ComplexF64,], i::Integer, j::Integer, d::Integer = 3)

Constructs the set `i`,`j`th Gell-Mann matrix of dimension `d`.

Reference: [Generalizations of Pauli matrices](https://en.wikipedia.org/wiki/Generalizations_of_Pauli_matrices)
"""
function gellmann(::Type{T}, i::Integer, j::Integer, d::Integer = 3) where {T<:Number}
    return gellmann!(zeros(T, d, d), i, j, d)
end
gellmann(i::Integer, j::Integer, d::Integer = 3) = gellmann(ComplexF64, i, j, d)

"""
    gellmann!(res::AbstractMatrix{T}, i::Integer, j::Integer, d::Integer = 3)

In-place version of `gellmann`.
"""
function gellmann!(res::AbstractMatrix{T}, i::Integer, j::Integer, d::Integer = 3) where {T<:Number}
    if i < j
        res[i, j] = 1
        res[j, i] = 1
    elseif i > j
        res[i, j] = im
        res[j, i] = -im
    elseif i == 1
        for k ∈ 1:d
            res[k, k] = 1 # _sqrt(T, 2) / _sqrt(T, d) if we want a proper normalisation
        end
    elseif i == d
        tmp = _sqrt(T, 2) / _sqrt(T, d * (d - 1))
        for k ∈ 1:(d-1)
            res[k, k] = tmp
        end
        res[d, d] = -(d - 1) * tmp
    else
        gellmann!(view(res, 1:(d-1), 1:(d-1)), i, j, d - 1)
    end
    return res
end
export gellmann!

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
    return sum(Hermitian(Ki * M * Ki') for Ki ∈ K)
end
export applykraus

function _orthonormal_range_svd!(
    A::AbstractMatrix{T};
    tol::Union{Real,Nothing} = nothing,
    alg = LinearAlgebra.default_svd_alg(A)
) where {T<:Number}
    dec = svd!(A; alg = alg)
    tol = isnothing(tol) ? maximum(dec.S) * _eps(T) * minimum(size(A)) : tol
    rank = sum(dec.S .> tol)
    dec.U[:, 1:rank]
end

_orthonormal_range_svd(A::AbstractMatrix; tol::Union{Real,Nothing} = nothing) =
    _orthonormal_range_svd!(deepcopy(A); tol)

function _orthonormal_range_qr(A::SA.AbstractSparseMatrix{T,M}; tol::Union{Real,Nothing} = nothing) where {T<:Number,M}
    dec = qr(A)
    tol = isnothing(tol) ? maximum(abs.(dec.R)) * _eps(T) : tol
    rank = sum(abs.(Diagonal(dec.R)) .> tol)
    SA.sparse(@view dec.Q[dec.rpivinv, 1:rank])
end

"""
    orthonormal_range(A::AbstractMatrix{T}; mode::Integer=nothing, tol::T=nothing, sp::Bool=true) where {T<:Number}

Orthonormal basis for the range of `A`. When `A` is sparse (or `mode = 0`), uses a QR factorization and returns a sparse result,
otherwise uses an SVD and returns a dense matrix (`mode = 1`). Input `A` will be overwritten during the factorization.
Tolerance `tol` is used to compute the rank and is automatically set if not provided.
"""
function orthonormal_range(
    A::SA.AbstractMatrix{T};
    mode::Integer = -1,
    tol::Union{Real,Nothing} = nothing
) where {T<:Number}
    mode == 1 && SA.issparse(A) && throw(ArgumentError("SVD does not work with sparse matrices, use a dense matrix."))
    mode == -1 && (mode = SA.issparse(A) ? 0 : 1)

    return (mode == 0 ? _orthonormal_range_qr(A; tol) : _orthonormal_range_svd(A; tol))
end
export orthonormal_range

"""
    symmetric_projection(dim::Integer, n::Integer; partial::Bool=true)

Orthogonal projection onto the symmetric subspace of `n` copies of a `dim`-dimensional space. By default (`partial=true`)
it returns an isometry (say, `V`) encoding the symmetric subspace. If `partial=false`, then it
returns the actual projection `V * V'`.

Reference: [Watrous' book](https://cs.uwaterloo.ca/~watrous/TQI/), Sec. 7.1.1
"""
function symmetric_projection(::Type{T}, dim::Integer, n::Integer; partial::Bool = true) where {T}
    is_sparse = T <: SA.CHOLMOD.VTypes #sparse qr decomposition fails for anything other than Float64 or ComplexF64
    P = is_sparse ? SA.spzeros(T, dim^n, dim^n) : zeros(T, dim^n, dim^n)
    perms = Combinatorics.permutations(1:n)
    for perm ∈ perms
        P .+= permutation_matrix(dim, perm; is_sparse)
    end
    P ./= length(perms)
    if partial
        V = orthonormal_range(P)
        size(V, 2) != binomial(n + dim - 1, dim - 1) && throw(AssertionError("Rank computation failed"))
        return V
    end
    return P
end
export symmetric_projection
symmetric_projection(dim::Integer, n::Integer; partial::Bool = true) = symmetric_projection(Float64, dim, n; partial)

"""
    n_body_basis(
    n::Integer,
    n_parties::Integer;
    sb::AbstractVector{<:AbstractMatrix} = [pauli(1), pauli(2), pauli(3)],
    sparse::Bool = true,
    eye::AbstractMatrix = I(size(sb[1], 1))

Return the basis of `n` nontrivial operators acting on `n_parties`, by default using Pauli matrices.

For example, `n_body_basis(2, 3)` generate all products of two Paulis and one identity, so
``{ X ⊗ X ⊗ 1, X ⊗ 1 ⊗ X, ..., X ⊗ Y ⊗ 1, ..., 1 ⊗ Z ⊗ Z}``.

Instead of Paulis, a basis can be provided by the parameter `sb`, and the identity can be changed with `eye`.
If `sparse` is true, the resulting basis will use sparse matrices, otherwise it will agree with `sb`.

This function returns a generator, which can then be used e.g. in for loops without fully allocating the
entire basis at once. If you need a vector, call `collect` on it.
"""
function n_body_basis(
    n::Integer,
    n_parties::Integer;
    sb::AbstractVector{<:AbstractMatrix} = [pauli(1), pauli(2), pauli(3)],
    sparse::Bool = true,
    eye::AbstractMatrix = I(size(sb[1], 1))
)
    (n >= 0 && n_parties >= 2) || throw(ArgumentError("Number of parties must be ≥ 2 and n ≥ 0."))
    n <= n_parties || throw(ArgumentError("Number of parties cannot be larger than n."))

    sb = sparse ? [SA.sparse.(sb); [eye]] : [sb; [eye]]
    nb = length(sb) - 1
    idx_eye = length(sb)
    (
        Base.kron(sb[t]...) for p ∈ Combinatorics.with_replacement_combinations(1:nb, n) for
        t ∈ Combinatorics.multiset_permutations([p; repeat([idx_eye], n_parties - n)], n_parties)
    )
end
export n_body_basis
