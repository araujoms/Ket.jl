"""
    permute_systems(X::AbstractVector, perm::Vector, dims::Vector)

Permutes the order of the subsystems of vector `X` with subsystem dimensions `dims` according to the permutation `perm`.
"""
function permute_systems(X::AbstractVector{T}, perm::Vector{<:Integer}, dims::Vector{<:Integer}) where {T}
    perm == 1:length(perm) && return X

    dX = length(X)
    dimsY = dims[perm]

    Y = Vector{T}(undef, dX)

    ti = Vector{Int64}(undef, length(dims))

    for i in 1:dX
        _tidx!(ti, i, dims)
        permute!(ti, perm)
        Yi = _idx(ti, dimsY)
        Y[Yi] = X[i]
    end
    return Y
end

"""
    permute_systems(X::AbstractMatrix, perm::Vector, dims::Vector)

Permutes the order of the subsystems of the square matrix `X`, which is composed by square subsystems of dimensions `dims`, according to the permutation `perm`.
"""
function permute_systems(X::AbstractMatrix, perm::Vector{<:Integer}, dims::Vector{<:Integer})
    perm == 1:length(perm) && return X

    idxperm = permute_systems(axes(X, 1), perm, dims)
    return X[idxperm, idxperm]
end

"""
    permute_systems(X::AbstractMatrix, perm::Vector, dims::Matrix)

Permutes the order of the subsystems of the matrix `X`, which is composed by subsystems of dimensions `dims`, according to the permutation `perm`.
`dims` should be a n x 2 matrix where `dims[i, 1]` is the number of rows of subsystem i, and `dims[i,2]` is its number of columns. 
"""
function permute_systems(X::AbstractMatrix, perm::Vector{<:Integer}, dims::Matrix{<:Integer})
    perm == 1:length(perm) && return X

    rowperm = permute_systems(axes(X, 1), perm, dims[:, 1])
    colperm = permute_systems(axes(X, 2), perm, dims[:, 2])

    return X[rowperm, colperm]
end
export permute_systems
