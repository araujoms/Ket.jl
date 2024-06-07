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

"""
    partial_trace(X::AbstractMatrix, remove::Vector, dims::Vector)

Takes the partial trace of matrix `X` with subsystem dimensions `dims` over the subsystems in `remove`.
"""
function partial_trace(X::AbstractMatrix{T}, remove::Vector{<:Integer}, dims::Vector{<:Integer}) where {T}
    isempty(remove) && return X
    length(remove) == length(dims) && return fill(T(LA.tr(X)), 1, 1)

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

    Y = Matrix{T}(undef, (dY, dY))       # Final output Y
    tXi = Vector{Int64}(undef, length(dims))    # Tensor indexing of X for column 
    tXj = Vector{Int64}(undef, length(dims))    # Tensor indexing of X for row

    @views tXikeep = tXi[keep]
    @views tXiremove = tXi[remove]
    @views tXjkeep = tXj[keep]
    @views tXjremove = tXj[remove]

    # We loop through Y and find the corresponding element
    @inbounds for i = 1:dY
        # Find current column tensor index for Y
        _tidx!(tXikeep, i, dimsY)
        for j = 1:dY
            # Find current row tensor index for Y
            _tidx!(tXjkeep, j, dimsY)

            # Now loop through the diagonal of the traced out systems 
            val = zero(T)
            for k = 1:dR
                _tidx!(tXiremove, k, dimsR)
                _tidx!(tXjremove, k, dimsR)

                # Find (i,j) index of X that we are currently on and add it to total
                Xi, Xj = _idx(tXi, dims), _idx(tXj, dims)
                val += X[Xi, Xj]
            end
            Y[i, j] = val
        end
    end
    return Y
end
"""
    partial_trace(X::AbstractMatrix, remove::Integer, dims::Vector)

Takes the partial trace of matrix `X` with subsystem dimensions `dims` over the subsystem `remove`.
"""
partial_trace(X::AbstractMatrix, remove::Integer, dims::Vector{<:Integer}) = partial_trace(X, [remove], dims)
export partial_trace
