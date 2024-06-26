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
    partial_trace(X::AbstractMatrix, remove::Vector, dims::Vector)

Takes the partial trace of matrix `X` with subsystem dimensions `dims` over the subsystems in `remove`.
""" partial_trace(X::AbstractMatrix, remove::Vector, dims::Vector)

for (T, limit, wrapper) in
    [(:AbstractMatrix, :dY, :identity), (:(LA.Hermitian), :j, :(LA.Hermitian)), (:(LA.Symmetric), :j, :(LA.Symmetric))]
    @eval begin
        function partial_trace(X::$T, remove::Vector{<:Integer}, dims::Vector{<:Integer})
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
"""
    partial_trace(X::AbstractMatrix, remove::Integer, dims::Vector)

Takes the partial trace of matrix `X` with subsystem dimensions `dims` over the subsystem `remove`.
"""
partial_trace(X::AbstractMatrix, remove::Integer, dims::Vector{<:Integer}) = partial_trace(X, [remove], dims)
export partial_trace

@doc """
    partial_transpose(X::AbstractMatrix, transp::Vector, dims::Vector)

Takes the partial transpose of matrix `X` with subsystem dimensions `dims` on the subsystems in `transp`.
""" partial_transpose(X::AbstractMatrix, transp::Vector, dims::Vector)

for (T, wrapper) in
    [(:AbstractMatrix, :identity), (:(LA.Hermitian), :(LA.Hermitian)), (:(LA.Symmetric), :(LA.Symmetric))]
    @eval begin
        function partial_transpose(X::$T, transp::Vector{<:Integer}, dims::Vector{<:Integer})
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
"""
    partial_transpose(X::AbstractMatrix, transp::Integer, dims::Vector)

Takes the partial transpose of matrix `X` with subsystem dimensions `dims` on the subsystem `transp`.
"""
partial_transpose(X::AbstractMatrix, transp::Integer, dims::Vector{<:Integer}) = partial_transpose(X, [transp], dims)
export partial_transpose
