"""
    local_bound(G::Array{T,4})

Computes the local bound of a bipartite Bell functional `G`, written in full probability notation
as a 4-dimensional array.

Reference: Araújo, Hirsch, and Quintino, [arXiv:2005.13418](https://arxiv.org/abs/2005.13418).
"""
function local_bound(G::Array{T, 4}) where {T <: Real}
    oa, ob, ia, ib = size(G)

    if oa^ia < ob^ib
        G = permutedims(G, (2, 1, 4, 3))
        oa, ob, ia, ib = size(G)
    end
    sizeG = size(G) #this is a workaround for julia issue #15276

    G = permutedims(G, (1, 3, 2, 4))
    squareG = reshape(G, oa * ia, ob * ib)
    offset = Vector(1 .+ ob * (0:(ib - 1)))
    @views initial_score = sum(maximum(reshape(sum(squareG[:, offset]; dims = 2), oa, ia); dims = 1)) #compute initial_score for the all-zeros strategy to serve as a reference point

    chunks = _partition(ob^ib - 1, Threads.nthreads())
    tasks = map(chunks) do chunk
        Threads.@spawn _local_bound_single(initial_score, chunk, sizeG, offset, squareG)
    end
    score = maximum(T.(fetch.(tasks))) #this type cast is to remove type instability

    return score
end
export local_bound

function _local_bound_single(initial_score::T, chunk, sizeG, offset, squareG::Array{T, 2}) where {T}
    oa, ob, ia, ib = sizeG
    score = initial_score
    ind = digits(chunk[1]; base = ob, pad = ib)
    offset_ind = zeros(Int64, ib)
    Galice = zeros(T, oa * ia, 1)
    maxvec = zeros(T, 1, ia)
    for b in chunk[1]:chunk[2]
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

If `n ≥ k` partitions the set `1:n` into `k` parts as equally sized as possible.
Otherwise partitions it into `n` parts of size 1.
"""
function _partition(n::T, k::T) where {T <: Integer}
    num_parts = min(k, n)
    parts = Vector{Tuple{T, T}}(undef, num_parts)
    base_size = div(n, k)
    num_larger = rem(n, k)
    if num_larger > 0
        parts[1] = (1, base_size + 1)
    else
        parts[1] = (1, base_size)
    end
    i = 2
    while i ≤ num_larger
        parts[i] = (1, base_size + 1) .+ parts[i - 1][2]
        i += 1
    end
    while i ≤ num_parts
        parts[i] = (1, base_size) .+ parts[i - 1][2]
        i += 1
    end
    return parts
end

# copied from QETLAB
function _update_odometer!(ind::AbstractVector{<:Integer}, upper_lim::Integer)
    # Start by increasing the last index by 1.
    ind_len = length(ind)
    ind[end] += 1

    # Now we work the "odometer": repeatedly set each digit to 0 if it
    # is too high and carry the addition to the left until we hit a
    # digit that *isn't* too high.
    for j in ind_len:-1:1
        # If we've hit the upper limit in this entry, move onto the next
        # entry.
        if ind[j] ≥ upper_lim
            ind[j] = 0
            if j ≥ 2
                ind[j - 1] += 1
            else # we're at the left end of the vector; just stop
                return
            end
        else
            return # always return if the odometer doesn't turn over
        end
    end
end

"""
    tsirelson_bound(CG::Matrix, scenario::Vector, level::Integer)

Upper bounds the Tsirelson bound of a bipartite Bell funcional game `CG`, written in Collins-Gisin notation.
`scenario` is a vector detailing the number of inputs and outputs, in the order [oa, ob, ia, ib].
`level` is an integer determining the level of the NPA hierarchy.

This function requires [Moment](https://github.com/ajpgarner/moment). It is only available if you first do "import MATLAB" or "using MATLAB".
"""
function tsirelson_bound(CG::Matrix{<:Real}, scenario::Vector{<:Integer}, level)
    error("This function requires MATLAB. Do `import MATLAB` or `using MATLAB` in order to enable it.")
end
export tsirelson_bound

"""
    tensor_collinsgisin(V::Array{T,4}, behaviour::Bool = false) where {T <: Real}

Takes a bipartite Bell functional `V` in full probability notation and transforms it to Collins-Gisin notation.
If `behaviour` is `true` do instead the transformation for behaviours. Doesn't assume normalization.
"""
function tensor_collinsgisin(V::AbstractArray{T, 4}, behaviour::Bool = false) where {T}
    oa, ob, ia, ib = size(V)
    alice_pars = ia * (oa - 1) + 1
    bob_pars = ib * (ob - 1) + 1
    aindex(a, x) = 1 + a + (x - 1) * (oa - 1)
    bindex(b, y) = 1 + b + (y - 1) * (ob - 1)

    CG = zeros(T, alice_pars, bob_pars)

    if ~behaviour
        CG[1, 1] = sum(V[oa, ob, :, :])
        for a in 1:(oa - 1), x in 1:ia
            CG[aindex(a, x), 1] = sum(V[a, ob, x, :] - V[oa, ob, x, :])
        end
        for b in 1:(ob - 1), y in 1:ib
            CG[1, bindex(b, y)] = sum(V[oa, b, :, y] - V[oa, ob, :, y])
        end
        for a in 1:(oa - 1), b in 1:(ob - 1), x in 1:ia, y in 1:ib
            CG[aindex(a, x), bindex(b, y)] = V[a, b, x, y] - V[a, ob, x, y] - V[oa, b, x, y] + V[oa, ob, x, y]
        end
    else
        CG[1, 1] = sum(V) / (ia * ib)
        for x in 1:ia, a in 1:(oa - 1)
            CG[aindex(a, x), 1] = sum(V[a, b, x, y] for b in 1:ob, y in 1:ib) / ib
        end
        for y in 1:ib, b in 1:(ob - 1)
            CG[1, bindex(b, y)] = sum(V[a, b, x, y] for a in 1:oa, x in 1:ia) / ia
        end
        for x in 1:ia, y in 1:ib
            CG[aindex(1, x):aindex(oa - 1, x), bindex(1, y):bindex(ob - 1, y)] = V[1:(oa - 1), 1:(ob - 1), x, y]
        end
    end
    return CG
end
export tensor_collinsgisin

"""
    tensor_probability(CG::Matrix, scenario::Vector, behaviour::Bool = false)

Takes a bipartite Bell functional `CG` in Collins-Gisin notation and transforms it to full probability notation.
`scenario` is a vector detailing the number of inputs and outputs, in the order [oa, ob, ia, ib].
If `behaviour` is `true` do instead the transformation for behaviours. Doesn't assume normalization.
"""
function tensor_probability(CG::AbstractMatrix{T}, scenario::Vector{<:Integer}, behaviour::Bool = false) where {T}
    oa, ob, ia, ib = scenario
    aindex(a, x) = 1 + a + (x - 1) * (oa - 1)
    bindex(b, y) = 1 + b + (y - 1) * (ob - 1)

    V = Array{T, 4}(undef, (oa, ob, ia, ib))

    if ~behaviour
        for x in 1:ia, y in 1:ib
            V[oa, ob, x, y] = CG[1, 1] / (ia * ib)
        end
        for x in 1:ia, y in 1:ib, b in 1:(ob - 1)
            V[oa, b, x, y] = CG[1, 1] / (ia * ib) + CG[1, bindex(b, y)] / ia
        end
        for x in 1:ia, y in 1:ib, a in 1:(oa - 1)
            V[a, ob, x, y] = CG[1, 1] / (ia * ib) + CG[aindex(a, x), 1] / ib
        end
        for x in 1:ia, y in 1:ib, a in 1:(oa - 1), b in 1:(ob - 1)
            V[a, b, x, y] =
                CG[1, 1] / (ia * ib) +
                CG[aindex(a, x), 1] / ib +
                CG[1, bindex(b, y)] / ia +
                CG[aindex(a, x), bindex(b, y)]
        end
    else
        for x in 1:ia, y in 1:ib
            V[1:(oa - 1), 1:(ob - 1), x, y] = CG[aindex(1, x):aindex(oa - 1, x), bindex(1, y):bindex(ob - 1, y)]
            V[1:(oa - 1), ob, x, y] =
                CG[aindex(1, x):aindex(oa - 1, x), 1] -
                sum(CG[aindex(1, x):aindex(oa - 1, x), bindex(1, y):bindex(ob - 1, y)]; dims = 2)
            V[oa, 1:(ob - 1), x, y] =
                CG[1, bindex(1, y):bindex(ob - 1, y)] -
                vec(sum(CG[aindex(1, x):aindex(oa - 1, x), bindex(1, y):bindex(ob - 1, y)]; dims = 1))
            V[oa, ob, x, y] =
                CG[1, 1] - sum(CG[aindex(1, x):aindex(oa - 1, x), 1]) -
                sum(CG[1, bindex(1, y):bindex(ob - 1, y)]) +
                sum(CG[aindex(1, x):aindex(oa - 1, x), bindex(1, y):bindex(ob - 1, y)])
        end
    end
    return V
end
export tensor_probability

"""
    tensor_probability(rho::Hermitian, all_Aax::Vector{Measurement}...)
    tensor_probability(rho::Hermitian, Aax::Measurement, N::Integer)

Applies N sets of measurements onto a state `rho` to form a probability array.
"""
function tensor_probability(
        rho::Hermitian{T1, Matrix{T1}},
        first_Aax::Vector{Measurement{T2}}, # needed so that T2 is not unbounded
        other_Aax::Vector{Measurement{T2}}...
    ) where {T1, T2}
    T = real(promote_type(T1, T2))
    all_Aax = (first_Aax, other_Aax...)
    N = length(all_Aax)
    m = length.(all_Aax) # numbers of inputs per party
    o = broadcast(Aax -> maximum(length.(Aax)), all_Aax) # numbers of outputs per party
    p = zeros(T, o..., m...)
    cia = CartesianIndices(o)
    cix = CartesianIndices(m)
    for a in cia, x in cix
        if all([a[n] ≤ length(all_Aax[n][x[n]]) for n in 1:N])
            p[a, x] = real(dot(Hermitian(kron([all_Aax[n][x[n]][a[n]] for n in 1:N]...)), rho))
        end
    end
    return p
end
# shorthand syntax for identical measurements on all parties
function tensor_probability(rho::Hermitian, Aax::Vector{<:Measurement}, N::Integer)
    return tensor_probability(rho, fill(Aax, N)...)
end
export tensor_probability

"""
    tensor_correlation(p::AbstractArray{T, N2}, behaviour::Bool = true; marg::Bool = true)

Converts a 2x...x2xmx...xm probability array into
- a mx...xm correlation array (no marginals)
- a (m+1)x...x(m+1) correlation array (marginals).
If `behaviour` is `true` do the transformation for behaviours. Doesn't assume normalization.

Also accepts the arguments of `tensor_probability` (state and measurements) for convenience.
"""
function tensor_correlation(p::AbstractArray{T, N2}, behaviour::Bool = true; marg::Bool = true) where {T} where {N2}
    @assert iseven(N2)
    N = N2 ÷ 2
    m = size(p)[(N + 1):end] # numbers of inputs per party
    o = size(p)[1:N] # numbers of outputs per party
    @assert collect(o) == 2ones(Int64, N)
    size_output = Tuple(marg ? m .+ 1 : m)
    res = zeros(T, size_output)
    cia = CartesianIndices(Tuple(2ones(Int64, N)))
    cix = CartesianIndices(size_output)
    for x in cix
        x_colon = Union{Colon, Int64}[x[n] > marg ? x[n] - marg : Colon() for n in 1:N]
        if ~behaviour
            error("Not implemented yet")
        else
            res[x] =
                sum((-1)^sum(a[n] for n in 1:N if x[n] > marg; init = 0) * sum(p[a, x_colon...]) for a in cia) /
                prod(m[n] for n in 1:N if x[n] == marg; init = 1)
        end
        if abs2(res[x]) < _eps(T)
            res[x] = 0
        end
    end
    return res
end
# accepts directly the arguments of tensor_probability
function tensor_correlation(rho::Hermitian, all_Aax::Vector{<:Measurement}...; marg::Bool = true)
    return tensor_correlation(tensor_probability(rho, all_Aax...), true; marg)
end
# shorthand syntax for identical measurements on all parties
function tensor_correlation(rho::Hermitian, Aax::Vector{<:Measurement}, N::Integer; marg::Bool = true)
    return tensor_correlation(rho, fill(Aax, N)...; marg)
end
export tensor_correlation
