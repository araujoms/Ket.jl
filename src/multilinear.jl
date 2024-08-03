"""
    _tidx(idx::Integer, dims::Vector)

Converts a standard index `idx` to a tensor index [i₁, i₂, ...] with subsystems dimensions `dims`.
"""
function _tidx(idx::Integer, dims::Vector{<:Integer})
    result = Vector{Int64}(undef, length(dims))
    _tidx!(result, idx, dims)
    return result
end

function _tidx!(tidx::AbstractVector{<:Integer}, idx::Integer, dims::Vector{<:Integer})
    nsys = length(dims)
    cidx = idx - 1          # Current index 
    dr = prod(dims)
    for k = 1:nsys
        # Everytime you increase a tensor index you shift by the product of remaining dimensions
        dr ÷= dims[k]
        tidx[k] = (cidx ÷ dr) + 1
        cidx %= dr
    end
    return tidx
end

"""
    _idx(tidx::Vector, dims::Vector)

Converts a tensor index `tidx` = [i₁, i₂, ...] with subsystems dimensions `dims` to a standard index.
"""
function _idx(tidx::Vector{<:Integer}, dims::Vector{<:Integer})
    i = 1
    shift = 1

    for k in length(tidx):-1:1
        i += (tidx[k] - 1) * shift
        shift *= dims[k]
    end
    return i
end

@doc """
    partial_trace(X::AbstractMatrix, remove::AbstractVector, dims::AbstractVector = _equal_sizes(X))

Takes the partial trace of matrix `X` with subsystem dimensions `dims` over the subsystems in `remove`. If the argument `dims` is omitted two equally-sized subsystems are assumed.
""" partial_trace(X::AbstractMatrix, remove::AbstractVector, dims::AbstractVector = _equal_sizes(X))

for (T, limit, wrapper) in
    [(:AbstractMatrix, :dY, :identity), (:(LA.Hermitian), :j, :(LA.Hermitian)), (:(LA.Symmetric), :j, :(LA.Symmetric))]
    @eval begin
        function partial_trace(
            X::$T,
            remove::AbstractVector{<:Integer},
            dims::AbstractVector{<:Integer} = _equal_sizes(X)
        )
            isempty(remove) && return X
            length(remove) == length(dims) && return fill(LA.tr(X), 1, 1)

            keep = Vector{eltype(remove)}(undef, length(dims) - length(remove))  # Systems kept 
            counter = 0
            for i = 1:length(dims)
                if !(i in remove)
                    counter += 1
                    keep[counter] = i
                end
            end
            dimsY = dims[keep]                      # The tensor dimensions of Y 
            dimsR = dims[remove]                          # The tensor dimensions of the traced out systems
            dY = prod(dimsY)                           # Dimension of Y
            dR = prod(dimsR)                           # Dimension of system traced out

            Y = similar(X, (dY, dY))       # Final output Y
            tXi = Vector{Int64}(undef, length(dims))    # Tensor indexing of X for column 
            tXj = Vector{Int64}(undef, length(dims))    # Tensor indexing of X for row

            @views tXikeep = tXi[keep]
            @views tXiremove = tXi[remove]
            @views tXjkeep = tXj[keep]
            @views tXjremove = tXj[remove]

            # We loop through Y and find the corresponding element
            @inbounds for j = 1:dY
                # Find current column tensor index for Y
                _tidx!(tXjkeep, j, dimsY)
                for i = 1:$limit
                    # Find current row tensor index for Y
                    _tidx!(tXikeep, i, dimsY)

                    # Now loop through the diagonal of the traced out systems 
                    Y[i, j] = 0
                    for k = 1:dR
                        _tidx!(tXiremove, k, dimsR)
                        _tidx!(tXjremove, k, dimsR)

                        # Find (i,j) index of X that we are currently on and add it to total
                        Xi, Xj = _idx(tXi, dims), _idx(tXj, dims)
                        Y[i, j] += X[Xi, Xj]
                    end
                end
            end
            if !isbits(Y[1]) #this is a workaround for a bug in Julia <= 1.10
                if $T == LA.Hermitian
                    LA.copytri!(Y, 'U', true)
                elseif $T == LA.Symmetric
                    LA.copytri!(Y, 'U')
                end
            end
            return $wrapper(Y)
        end
    end
end
export partial_trace

"""
    partial_trace(X::AbstractMatrix, remove::Integer, dims::AbstractVector = _equal_sizes(X)))

Takes the partial trace of matrix `X` with subsystem dimensions `dims` over the subsystem `remove`. If the argument `dims` is omitted two equally-sized subsystems are assumed.
"""
partial_trace(X::AbstractMatrix, remove::Integer, dims::AbstractVector{<:Integer} = _equal_sizes(X)) =
    partial_trace(X, [remove], dims)

@doc """
    partial_transpose(X::AbstractMatrix, transp::AbstractVector, dims::AbstractVector = _equal_sizes(X))

Takes the partial transpose of matrix `X` with subsystem dimensions `dims` on the subsystems in `transp`. If the argument `dims` is omitted two equally-sized subsystems are assumed.
""" partial_transpose(X::AbstractMatrix, transp::AbstractVector, dims::AbstractVector = _equal_sizes(X))

for (T, wrapper) in
    [(:AbstractMatrix, :identity), (:(LA.Hermitian), :(LA.Hermitian)), (:(LA.Symmetric), :(LA.Symmetric))]
    @eval begin
        function partial_transpose(
            X::$T,
            transp::AbstractVector{<:Integer},
            dims::AbstractVector{<:Integer} = _equal_sizes(X)
        )
            isempty(transp) && return X
            length(transp) == length(dims) && return LA.transpose(X)

            keep = Vector{eltype(transp)}(undef, length(dims) - length(transp))  # Systems kept 
            counter = 0
            for i = 1:length(dims)
                if !(i in transp)
                    counter += 1
                    keep[counter] = i
                end
            end

            d = size(X, 1)                              # Dimension of the final output Y
            Y = similar(X, (d, d))                      # Final output Y

            tXi = Vector{Int64}(undef, length(dims))    # Tensor indexing of X for row 
            tXj = Vector{Int64}(undef, length(dims))    # Tensor indexing of X for column

            tYi = Vector{Int64}(undef, length(dims))    # Tensor indexing of Y for row 
            tYj = Vector{Int64}(undef, length(dims))    # Tensor indexing of Y for column

            @inbounds for j in 1:d
                _tidx!(tYj, j, dims)
                for i in 1:j-1
                    _tidx!(tYi, i, dims)

                    for k in keep
                        tXi[k] = tYi[k]
                        tXj[k] = tYj[k]
                    end

                    for t in transp
                        tXi[t] = tYj[t]
                        tXj[t] = tYi[t]
                    end

                    Xi, Xj = _idx(tXi, dims), _idx(tXj, dims)
                    Y[i, j] = X[Xi, Xj]
                    Y[j, i] = X[Xj, Xi]
                end
                for k in keep
                    tXj[k] = tYj[k]
                end

                for t in transp
                    tXj[t] = tYj[t]
                end

                Xj = _idx(tXj, dims)
                Y[j, j] = X[Xj, Xj]
            end
            return $wrapper(Y)
        end
    end
end
export partial_transpose

"""
    partial_transpose(X::AbstractMatrix, transp::Integer, dims::AbstractVector = _equal_sizes(X))

Takes the partial transpose of matrix `X` with subsystem dimensions `dims` on the subsystem `transp`. If the argument `dims` is omitted two equally-sized subsystems are assumed.
"""
partial_transpose(X::AbstractMatrix, transp::Integer, dims::AbstractVector{<:Integer} = _equal_sizes(X)) =
    partial_transpose(X, [transp], dims)

"""
    _idxperm(perm::Vector, dims::Vector)

Computes the index permutation associated with permuting the subsystems of a vector with subsystem dimensions `dims` according to `perm`.
"""
function _idxperm(perm::Vector{<:Integer}, dims::Vector{<:Integer})
    p = Vector{Int64}(undef, prod(dims))
    _idxperm!(p, perm, dims)
    return p
end

function _idxperm!(p::Vector{<:Integer}, perm::Vector{<:Integer}, dims::Vector{<:Integer})
    pdims = dims[perm]

    ti = Vector{Int64}(undef, length(dims))

    for i in eachindex(p)
        _tidx!(ti, i, dims)
        permute!(ti, perm)
        j = _idx(ti, pdims)
        p[j] = i
    end
    return p
end

"""
    permute_systems!(X::AbstractVector, perm::AbstractVector, dims::AbstractVector = _equal_sizes(X))

Permutes the order of the subsystems of vector `X` with subsystem dimensions `dims` in-place according to the permutation `perm`. If the argument `dims` is omitted two equally-sized subsystems are assumed.
"""
function permute_systems!(
    X::AbstractVector{T},
    perm::AbstractVector{<:Integer},
    dims::AbstractVector{<:Integer} = _equal_sizes(X)
) where {T}
    perm == 1:length(perm) && return X
    X == 1:length(X) && return _idxperm!(X, perm, dims)

    p = _idxperm(perm, dims)
    permute!(X, p)
    return X
end
export permute_systems!

@doc """
     permute_systems(X::AbstractMatrix, perm::AbstractVector, dims::AbstractVector = _equal_sizes(X))

 Permutes the order of the subsystems of the square matrix `X`, which is composed by square subsystems of dimensions `dims`, according to the permutation `perm`. If the argument `dims` is omitted two equally-sized subsystems are assumed.
 """ permute_systems(
    X::AbstractMatrix,
    perm::AbstractVector,
    dims::AbstractVector = _equal_sizes(X);
    rows_only::Bool = false
)
for (T, wrapper) in
    [(:AbstractMatrix, :identity), (:(LA.Hermitian), :(LA.Hermitian)), (:(LA.Symmetric), :(LA.Symmetric))]
    @eval begin
        function permute_systems(
            X::$T,
            perm::AbstractVector{<:Integer},
            dims::AbstractVector{<:Integer} = _equal_sizes(X);
            rows_only::Bool = false
        )
            perm == 1:length(perm) && return X

            p = _idxperm(perm, dims)
            return rows_only ? $wrapper(X[p, 1:end]) : $wrapper(X[p, p])
        end
    end
end

"""
    permute_systems(X::AbstractMatrix, perm::Vector, dims::Matrix)

Permutes the order of the subsystems of the matrix `X`, which is composed by subsystems of dimensions `dims`, according to the permutation `perm`.
`dims` should be a n x 2 matrix where `dims[i, 1]` is the number of rows of subsystem i, and `dims[i,2]` is its number of columns. 
"""
function permute_systems(X::AbstractMatrix, perm::AbstractVector{<:Integer}, dims::Matrix{<:Integer})
    perm == 1:length(perm) && return X

    rowp = _idxperm(perm, dims[:, 1])
    colp = _idxperm(perm, dims[:, 2])
    return X[rowp, colp]
end
export permute_systems

"""
    permutation_matrix(dims::Union{Integer,AbstractVector}, perm::AbstractVector)

Unitary that permutes subsystems of dimension `dims` according to the permutation `perm`.
If `dims` is an Integer, assumes there are `length(perm)` subsystems of equal dimensions `dims`.
"""
function permutation_matrix(
    dims::Union{Integer,AbstractVector{<:Integer}},
    perm::AbstractVector{<:Integer};
    sp::Bool = true
)
    dims = dims isa Integer ? fill(dims, length(perm)) : dims
    permute_systems(LA.I(prod(dims), sp), perm, dims; rows_only = true)
end
export permutation_matrix
