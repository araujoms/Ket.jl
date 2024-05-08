using LinearAlgebra
using InvertedIndices

export ptr

function tidx(idx::Integer, dims::Vector{<:Integer})
    #= 
    Converts a standard index i to a tensor index i1,i2,i3,...
    Parameters:
        idx     -       index
        dims    -       list of dimensions of the Hilbert spaces in tensor product
    Returns:
        tidx    -       tensor index [i1,i2,i3]
    Notes:
        Basically converting multipartite agnostic index |i> to index taking
        into account the tensor product structure |i1 i2 i3 ... > 

        To comply with Julia's count from 1 we also index from 1!
        (But in the actual computations we shift to 0 counting and then back again)
    =#
    nsys = length(dims)
    tidx = Vector{Int64}(undef, nsys)
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

function idx(tidx::Vector{<:Integer}, dims::Vector{<:Integer})
    #= 
    Converts a tensor index i1,i2,i3,... to a standard index i
    Parameters:
        tidx     -      tensor index i1,i2,i3,...
        dims    -       list of dimensions of the Hilbert spaces in tensor product
    Returns:
        idx    -       standard index i
    Notes:
        To comply with Julia's count from 1 we also index from 1!
        (But in the actual computations we shift to 0 counting and then back again)
    =#
    # nsys = length(dims);
    # idx = 1;
    # dr = prod(dims);
    # for k = 1:nsys
    #     dr = dr ÷ dims[k]
    #     idx += (tidx[k]-1) * dr
    # end
    # return idx
    i = 1
    shift = 1

    for k in length(tidx):-1:1
        i += (tidx[k] - 1) * shift
        shift *= dims[k]
    end
    return i
end

function ptr(X::AbstractMatrix, sys::Vector{<:Integer}, dims::Vector{<:Integer})
    #= 
    Traces out systems indicated by ksys. 
    Parameters:
        X       -       Square matrix of appropriate size 
        ksys    -       list of {0,1} with 1 indicating the corresponding system is traced out
        dims    -       list of dimensions of the systems 
    Returns:
        Y    -       Matrix which is partial trace of X
    =#

    sysidx = 1:length(dims)
    sysKeep = sysidx[Not(sys)]                  # Systems kept 
    dimsY = dims[sysKeep]                      # The tensor dimensions of Y 
    dimsR = dims[sys]                          # The tensor dimensions of the traced out systems
    dY = prod(dimsY)                           # Dimension of Y
    dR = prod(dimsR)                           # Dimension of system traced out

    Y = Matrix{eltype(X)}(undef, (dY, dY))       # Final output Y
    tXi = Vector{Int64}(undef, length(dims))    # Tensor indexing of X for column 
    tXj = Vector{Int64}(undef, length(dims))    # Tensor indexing of X for row

    # We loop through Y and find the corresponding element
    for i = 1:dY
        # Find current column tensor index for Y 
        ti = tidx(i, dimsY)
        tXi[sysKeep] = ti
        for j = 1:dY
            # Find current row tensor index for Y
            tj = tidx(j, dimsY)
            tXj[sysKeep] = tj
            # Now loop through the diagonal of the traced out systems 
            val = 0
            for k = 1:dR
                tk = tidx(k, dimsR)
                tXi[sys], tXj[sys] = tk, tk

                # Find (i,j) index of X that we are currently on and add it to total
                Xi, Xj = idx(tXi, dims), idx(tXj, dims)
                val += X[Xi, Xj]
            end
            Y[i, j] = val
        end
    end
    return Y
end
