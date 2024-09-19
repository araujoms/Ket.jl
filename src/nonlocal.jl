"""
    local_bound(G::Array{T,4})

Computes the local bound of a bipartite Bell functional `G`, written in full probability notation
as a 4-dimensional array.

Reference: Araújo, Hirsch, and Quintino, [arXiv:2005.13418](https://arxiv.org/abs/2005.13418).
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

    chunks = _partition(ob^ib - 1, Threads.nthreads())
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

If `n ≥ k` partitions the set `1:n` into `k` parts as equally sized as possible.
Otherwise partitions it into `n` parts of size 1.
"""
function _partition(n::T, k::T) where {T<:Integer}
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
    while i ≤ num_larger
        parts[i] = (1, base_size + 1) .+ parts[i-1][2]
        i += 1
    end
    while i ≤ num_parts
        parts[i] = (1, base_size) .+ parts[i-1][2]
        i += 1
    end
    return parts
end

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
        if ind[j] ≥ upper_lim
            ind[j] = 0
            if j ≥ 2
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
    tsirelson_bound(CG::Matrix, scenario::Vector, level::Integer)

Upper bounds the Tsirelson bound of a bipartite Bell funcional game `CG`, written in Collins-Gisin notation.
`scenario` is vector detailing the number of inputs and outputs, in the order [oa, ob, ia, ib].
`level` is an integer determining the level of the NPA hierarchy.

This function requires [Moment](https://github.com/ajpgarner/moment). It is only available if you first do "import MATLAB" or "using MATLAB".
"""
function tsirelson_bound(CG::Matrix{<:Real}, scenario::Vector{<:Integer}, level)
    error("This function requires MATLAB. Do `import MATLAB` or `using MATLAB` in order to enable it.")
end
export tsirelson_bound

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

"""
    probability_tensor(rho::LA.Hermitian, all_Aax::Vector{Measurement}...)

Applies N sets of measurements onto a state `rho` to form a probability array.
"""
function probability_tensor(
    rho::LA.Hermitian{T1,Matrix{T1}},
    first_Aax::Vector{Measurement{T2}}, # needed so that T2 is not unbounded
    other_Aax::Vector{Measurement{T2}}...
) where {T1,T2}
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
            p[a, x] = real(LA.dot(LA.Hermitian(kron([all_Aax[n][x[n]][a[n]] for n in 1:N]...)), rho))
        end
    end
    return p
end
# accepts a pure state
function probability_tensor(psi::AbstractVector, all_Aax::Vector{<:Measurement}...)
    return probability_tensor(ketbra(psi), all_Aax...)
end
# accepts projective measurements
function probability_tensor(rho::LA.Hermitian, all_φax::Vector{<:AbstractMatrix}...)
    return probability_tensor(rho, povm.(all_φax)...)
end
# accepts pure states and projective measurements
function probability_tensor(psi::AbstractVector, all_φax::Vector{<:AbstractMatrix}...)
    return probability_tensor(ketbra(psi), povm.(all_φax)...)
end
# shorthand syntax for identical measurements on all parties
function probability_tensor(rho::LA.Hermitian, Aax::Vector{<:Measurement}, N::Integer)
    return probability_tensor(rho, fill(Aax, N)...)
end
function probability_tensor(psi::AbstractVector, Aax::Vector{<:Measurement}, N::Integer)
    return probability_tensor(psi, fill(Aax, N)...)
end
function probability_tensor(rho::LA.Hermitian, φax::Vector{<:AbstractMatrix}, N::Integer)
    return probability_tensor(rho, fill(povm(φax), N)...)
end
function probability_tensor(psi::AbstractVector, φax::Vector{<:AbstractMatrix}, N::Integer)
    return probability_tensor(psi, fill(povm(φax), N)...)
end
export probability_tensor

"""
    correlation_tensor(p::AbstractArray{T, N2}; marg::Bool = true)

Applies N sets of measurements onto a state `rho` to form a probability array.
Convert a 2x...x2xmx...xm probability array into
- a mx...xm correlation array (no marginals)
- a (m+1)x...x(m+1) correlation array (marginals).
"""
function correlation_tensor(p::AbstractArray{T,N2}; marg::Bool = true) where {T} where {N2}
    @assert iseven(N2)
    N = N2 ÷ 2
    m = size(p)[N+1:end] # numbers of inputs per party
    o = size(p)[1:N] # numbers of outputs per party
    @assert collect(o) == 2ones(Int64, N)
    res = zeros(T, (marg ? m .+ 1 : m)...)
    cia = CartesianIndices(Tuple(2ones(Int64, N)))
    cix = CartesianIndices(Tuple(marg ? m .+ 1 : m))
    for x in cix
        x_colon = [x[n] ≤ m[n] ? x[n] : Colon() for n in 1:N]
        res[x] =
            sum((-1)^sum(a[n] for n in 1:N if x[n] ≤ m[n]; init = 0) * sum(p[a, x_colon...]) for a in cia) /
            prod(m[n] for n in 1:N if x[n] > m[n]; init = 1)
        if abs2(res[x]) < _eps(T)
            res[x] = 0
        end
    end
    return res
end
# accepts directly the arguments of probability_tensor
# SD: I'm still unsure whether it would be better practice to have a general syntax for this kind of argument passing
function correlation_tensor(rho::LA.Hermitian, all_Aax::Vector{<:Measurement}...; marg::Bool = true)
    return correlation_tensor(probability_tensor(rho, all_Aax...); marg)
end
function correlation_tensor(psi::AbstractVector, all_Aax::Vector{<:Measurement}...; marg::Bool = true)
    return correlation_tensor(probability_tensor(psi, all_Aax...); marg)
end
function correlation_tensor(rho::LA.Hermitian, all_φax::Vector{<:AbstractMatrix}...; marg::Bool = true)
    return correlation_tensor(probability_tensor(rho, all_φax...); marg)
end
function correlation_tensor(psi::AbstractVector, all_φax::Vector{<:AbstractMatrix}...; marg::Bool = true)
    return correlation_tensor(probability_tensor(psi, all_φax...); marg)
end
# shorthand syntax for identical measurements on all parties
function correlation_tensor(rho::LA.Hermitian, Aax::Vector{<:Measurement}, N::Integer; marg::Bool = true)
    return correlation_tensor(rho, fill(Aax, N)...; marg)
end
function correlation_tensor(psi::AbstractVector, Aax::Vector{<:Measurement}, N::Integer; marg::Bool = true)
    return correlation_tensor(psi, fill(Aax, N)...; marg)
end
function correlation_tensor(rho::LA.Hermitian, φax::Vector{<:AbstractMatrix}, N::Integer; marg::Bool = true)
    return correlation_tensor(rho, fill(povm(φax), N)...; marg)
end
function correlation_tensor(psi::AbstractVector, φax::Vector{<:AbstractMatrix}, N::Integer; marg::Bool = true)
    return correlation_tensor(psi, fill(povm(φax), N)...; marg)
end
export correlation_tensor
