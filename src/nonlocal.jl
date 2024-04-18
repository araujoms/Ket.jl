"""
    local_bound(G::Array{T,4})
    
Computes the local bound of a bipartite Bell functional `G`, written in full probability notation
as a 4-dimensional array.
"""
function local_bound(G::Array{T,4}) where {T<:Real}
    oa, ob, ia, ib = size(G)

    if oa^ia < ob^ib
        G = permutedims(G, (2, 1, 4, 3))
        oa, ob, ia, ib = size(G)
    end
    sizeG = size(G) #this is a workaround for julia issue #15276

    G = permutedims(G, (1, 3, 2, 4))
    squareG = reshape(G, oa * ia, ob * ib)
    offset = Vector(1 .+ ob * (0:ib-1))
    @views initial_score = sum(maximum(reshape(sum(squareG[:, offset]; dims = 2), oa, ia); dims = 1)) #compute initial_score for the all-zeros strategy to serve as a reference point

    chunks = partition(ob^ib - 1, Threads.nthreads())
    tasks = map(chunks) do chunk
        Threads.@spawn _local_bound_single(initial_score, chunk, sizeG, offset, squareG)
    end
    score = maximum(T.(fetch.(tasks))) #this type cast is to remove type instability

    return score
end
export local_bound

function _local_bound_single(initial_score::T, chunk, sizeG, offset, squareG::Array{T,2}) where {T}
    oa, ob, ia, ib = sizeG
    score = initial_score
    ind = digits(chunk[1]; base = ob, pad = ib)
    offset_ind = zeros(Int64, ib)
    Galice = zeros(T, oa * ia, 1)
    maxvec = zeros(T, 1, ia)
    for b = chunk[1]:chunk[2]
        offset_ind .= ind .+ offset
        @views sum!(Galice, squareG[:, offset_ind])
        squareGalice = Base.ReshapedArray(Galice, (oa, ia), ())
        temp_score = sum(maximum!(maxvec, squareGalice))
        score = max(score, temp_score)
        _update_odometer!(ind, ob)
    end

    return score
end

"""
    partition(n::Integer, k::Integer)

If `n >= k` partitions the set `1:n` into `k` parts as equally sized as possible.
Otherwise partitions it into `n` parts of size 1."""
function partition(n::T, k::T) where {T<:Integer}
    num_parts = min(k, n)
    parts = Vector{Tuple{T,T}}(undef, num_parts)
    base_size = div(n, k)
    num_larger = rem(n, k)
    if num_larger > 0
        parts[1] = (1, base_size + 1)
    else
        parts[1] = (1, base_size)
    end
    i = 2
    while i <= num_larger
        parts[i] = (1, base_size + 1) .+ parts[i-1][2]
        i += 1
    end
    while i <= num_parts
        parts[i] = (1, base_size) .+ parts[i-1][2]
        i += 1
    end
    return parts
end
export partition

#copyed from QETLAB
function _update_odometer!(ind::AbstractVector{<:Integer}, upper_lim::Integer)
    # Start by increasing the last index by 1.
    ind_len = length(ind)
    ind[end] += 1

    # Now we work the "odometer": repeatedly set each digit to 0 if it
    # is too high and carry the addition to the left until we hit a
    # digit that *isn't* too high.
    for j = ind_len:-1:1
        # If we've hit the upper limit in this entry, move onto the next
        # entry.
        if ind[j] >= upper_lim
            ind[j] = 0
            if j >= 2
                ind[j-1] += 1
            else # we're at the left end of the vector; just stop
                return
            end
        else
            return # always return if the odometer doesn't turn over
        end
    end
end

"""
    fp2cg(V::Array{T,4}) where {T <: Real}

Takes a bipartite Bell functional `V` in full probability notation and transforms it
to Collins-Gisin notation.
"""
function fp2cg(V::AbstractArray{T,4}) where {T<:Real}
    oa, ob, ia, ib = size(V)
    alice_pars = ia * (oa - 1) + 1
    bob_pars = ib * (ob - 1) + 1
    aindex(a, x) = 1 + a + (x - 1) * (oa - 1)
    bindex(b, y) = 1 + b + (y - 1) * (ob - 1)

    CG = zeros(T, alice_pars, bob_pars)

    CG[1, 1] = sum(V[oa, ob, :, :])
    for a = 1:oa-1, x = 1:ia
        CG[aindex(a, x), 1] = sum(V[a, ob, x, :] - V[oa, ob, x, :])
    end
    for b = 1:ob-1, y = 1:ib
        CG[1, bindex(b, y)] = sum(V[oa, b, :, y] - V[oa, ob, :, y])
    end
    for a = 1:oa-1, b = 1:ob-1, x = 1:ia, y = 1:ib
        CG[aindex(a, x), bindex(b, y)] = V[a, b, x, y] - V[a, ob, x, y] - V[oa, b, x, y] + V[oa, ob, x, y]
    end

    return CG
end
export fp2cg
