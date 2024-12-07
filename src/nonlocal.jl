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
    return
end

"""
    tsirelson_bound(CG::Matrix, scenario::Vector, level::Integer)

Upper bounds the Tsirelson bound of a bipartite Bell funcional game `CG`, written in Collins-Gisin notation.
`scenario` is a vector detailing the number of inputs and outputs, in the order [oa, ob, ia, ib].
`level` is an integer determining the level of the NPA hierarchy.

This function requires [Moment](https://github.com/ajpgarner/moment). It is only available if you first do "import MATLAB" or "using MATLAB".
"""
function tsirelson_bound(CG::Matrix{<:Real}, scenario::Vector{<:Integer}, level)
    return error("This function requires MATLAB. Do `import MATLAB` or `using MATLAB` in order to enable it.")
end
export tsirelson_bound

"""
    tensor_collinsgisin(V::Array{T,4}, behaviour::Bool = false)

Takes a bipartite Bell functional `V` in full probability notation and transforms it to Collins-Gisin notation.
If `behaviour` is `true` do instead the transformation for behaviours. Doesn't assume normalization.

Also accepts the arguments of `tensor_probability` (state and measurements) for convenience.
"""
function tensor_collinsgisin(p::AbstractArray{T, N2}, behaviour::Bool = false) where {T, N2}
    @assert iseven(N2)
    N = N2 ÷ 2

    if ~behaviour
        N == 2 || error("Multipartite transformation for functionals not yet implemented.")
        oa, ob, ia, ib = size(p)
        alice_pars = ia * (oa - 1) + 1
        bob_pars = ib * (ob - 1) + 1
        aindex(a, x) = 1 + a + (x - 1) * (oa - 1)
        bindex(b, y) = 1 + b + (y - 1) * (ob - 1)
        CG = zeros(T, alice_pars, bob_pars)
        CG[1, 1] = sum(p[oa, ob, :, :])
        for a in 1:(oa - 1), x in 1:ia
            CG[aindex(a, x), 1] = sum(p[a, ob, x, :] - p[oa, ob, x, :])
        end
        for b in 1:(ob - 1), y in 1:ib
            CG[1, bindex(b, y)] = sum(p[oa, b, :, y] - p[oa, ob, :, y])
        end
        for a in 1:(oa - 1), b in 1:(ob - 1), x in 1:ia, y in 1:ib
            CG[aindex(a, x), bindex(b, y)] = p[a, b, x, y] - p[a, ob, x, y] - p[oa, b, x, y] + p[oa, ob, x, y]
        end
    else
        scenario = size(p)
        outs = scenario[1:N]
        num_outs = prod(outs)

        ins = scenario[(N + 1):(2 * N)]
        num_ins = prod(ins)

        p2cg(a, x) = (a .!= outs) .* (a + (x .- 1) .* (outs .- 1)) .+ 1

        cgdesc = ins .* (outs .- 1) .+ 1
        cgprodsizes = ones(Int, N)
        for i in 1:N
            cgprodsizes[i] = prod(cgdesc[1:(i - 1)])
        end
        cgindex(posvec) = cgprodsizes' * (posvec .- 1) + 1
        prodsizes = ones(Int, 2 * N)
        for i in 1:(2 * N)
            prodsizes[i] = prod(scenario[1:(i - 1)])
        end
        pindex(posvec) = prodsizes' * (posvec .- 1) + 1
        CG = zeros(T, cgdesc...)

        for inscalar in 0:(num_ins - 1)
            invec = 1 .+ _digits_mixed_basis(inscalar, ins)
            for outscalar in 0:(num_outs - 1)
                outvec = 1 .+ _digits_mixed_basis(outscalar, outs)
                for outscalar2 in 0:(num_outs - 1)
                    outvec2 = 1 .+ _digits_mixed_basis(outscalar2, outs)
                    if (outvec .!= outs) .* outvec == (outvec .!= outs) .* outvec2
                        CG[cgindex(p2cg(outvec, invec))] += p[pindex([outvec2; invec])] / prod(ins[outvec .== outs])
                    end
                end
            end
        end
    end
    return CG
end
# accepts directly the arguments of tensor_probability
function tensor_collinsgisin(rho::Hermitian, all_Aax::Vector{<:Measurement}...)
    return tensor_collinsgisin(tensor_probability(rho, all_Aax...), true)
end
# shorthand syntax for identical measurements on all parties
function tensor_collinsgisin(rho::Hermitian, Aax::Vector{<:Measurement}, N::Integer)
    return tensor_collinsgisin(rho, fill(Aax, N)...)
end
export tensor_collinsgisin

"""
    tensor_probability(CG::Array{T, N}, scenario::Vector, behaviour::Bool = false)

Takes a bipartite Bell functional `CG` in Collins-Gisin notation and transforms it to full probability notation.
`scenario` is a vector detailing the number of inputs and outputs, in the order [oa, ob, ia, ib].
If `behaviour` is `true` do instead the transformation for behaviours. Doesn't assume normalization.
"""
function tensor_probability(CG::AbstractArray{T, N}, scenario::Vector{<:Integer}, behaviour::Bool = false) where {T, N}
    p = zeros(T, scenario...)

    if ~behaviour
        N == 2 || error("Multipartite transformation for functionals not yet implemented.")
        oa, ob, ia, ib = scenario
        aindex(a, x) = 1 + a + (x - 1) * (oa - 1)
        bindex(b, y) = 1 + b + (y - 1) * (ob - 1)
        for x in 1:ia, y in 1:ib
            p[oa, ob, x, y] = CG[1, 1] / (ia * ib)
        end
        for x in 1:ia, y in 1:ib, b in 1:(ob - 1)
            p[oa, b, x, y] = CG[1, 1] / (ia * ib) + CG[1, bindex(b, y)] / ia
        end
        for x in 1:ia, y in 1:ib, a in 1:(oa - 1)
            p[a, ob, x, y] = CG[1, 1] / (ia * ib) + CG[aindex(a, x), 1] / ib
        end
        for x in 1:ia, y in 1:ib, a in 1:(oa - 1), b in 1:(ob - 1)
            p[a, b, x, y] = CG[1, 1] / (ia * ib) + CG[aindex(a, x), 1] / ib + CG[1, bindex(b, y)] / ia + CG[aindex(a, x), bindex(b, y)]
        end
    else
        outs = scenario[1:N]
        num_outs = prod(outs)

        ins = scenario[(N + 1):(2 * N)]
        num_ins = prod(ins)

        p2cg(a, x) = (a .!= outs) .* (a + (x .- 1) .* (outs .- 1)) .+ 1

        cgdesc = size(CG)
        cgprodsizes = ones(Int, N)
        for i in 1:N
            cgprodsizes[i] = prod(cgdesc[1:(i - 1)])
        end
        cgindex(posvec) = cgprodsizes' * (posvec .- 1) + 1

        prodsizes = ones(Int, 2 * N)
        for i in 1:(2 * N)
            prodsizes[i] = prod(scenario[1:(i - 1)])
        end
        pindex(posvec) = prodsizes' * (posvec .- 1) + 1

        for inscalar in 0:(num_ins - 1)
            invec = 1 .+ _digits_mixed_basis(inscalar, ins)
            for outscalar in 0:(num_outs - 1)
                outvec = 1 .+ _digits_mixed_basis(outscalar, outs)
                for outscalar2 in 0:(num_outs - 1)
                    outvec2 = 1 .+ _digits_mixed_basis(outscalar2, outs)
                    if (outvec .!= outs) .* outvec == (outvec .!= outs) .* outvec2
                        ndiff = abs(sum(outvec .!= outs) - sum(outvec2 .!= outs))
                        p[pindex([outvec; invec])] += (-1)^ndiff * CG[cgindex(p2cg(outvec2, invec))]
                    end
                end
            end
        end
    end
    return p
end

function _digits_mixed_basis(ind, bases)
    N = length(bases)
    digits = zeros(Int, N)
    for i in N:-1:1
        digits[i] = mod(ind, bases[i])
        ind = div(ind, bases[i])
    end
    return digits
end

"""
    tensor_probability(FC::Matrix, behaviour::Bool = false)

Takes a bipartite Bell functional `FC` in full correlator notation and transforms it to full probability notation.
If `behaviour` is `true` do instead the transformation for behaviours. Doesn't assume normalization.
"""
function tensor_probability(FC::AbstractArray{T, N}, behaviour::Bool = false) where {T, N}
    o = Tuple(fill(2, N))
    m = size(FC) .- 1
    FP = zeros(T, o..., m...)
    cia = CartesianIndices(o)
    cix = CartesianIndices(m)
    # there may be a smarter way to order these loops
    for a2 in cia
        ind = collect(a2.I) .== 2
        denominator = behaviour ? 1 : prod(m[.~ind]; init = 1)
        for a1 in cia
            s = (-1)^sum(a1.I[ind] .- 1; init = 0)
            for x in cix
                FP[a1, x] += s * FC[[a2[n] == 1 ? 1 : x[n] + 1 for n in 1:N]...] / denominator
            end
        end
    end
    if behaviour
        FP ./= 2^N
    end
    cleanup!(FP)
    return FP
end

"""
    tensor_probability(rho::Hermitian, all_Aax::Vector{Measurement}...)
    tensor_probability(rho::Hermitian, Aax::Vector{Measurement}, N::Integer)

Applies N sets of measurements onto a state `rho` to form a probability array.
If all parties apply the same measurements, use the shorthand notation.
"""
function tensor_probability(
        rho::Hermitian{T1, Matrix{T1}},
        first_Aax::Vector{Measurement{T2}}, # needed so that T2 is not unbounded
        other_Aax::Vector{Measurement{T2}}...,
    ) where {T1, T2}
    T = real(promote_type(T1, T2))
    all_Aax = (first_Aax, other_Aax...)
    N = length(all_Aax)
    m = length.(all_Aax) # numbers of inputs per party
    o = broadcast(Aax -> maximum(length.(Aax)), all_Aax) # numbers of outputs per party
    FP = zeros(T, o..., m...)
    cia = CartesianIndices(o)
    cix = CartesianIndices(m)
    for a in cia, x in cix
        if all([a[n] ≤ length(all_Aax[n][x[n]]) for n in 1:N])
            FP[a, x] = real(dot(Hermitian(kron([all_Aax[n][x[n]][a[n]] for n in 1:N]...)), rho))
        end
    end
    return FP
end
# shorthand syntax for identical measurements on all parties
function tensor_probability(rho::Hermitian, Aax::Vector{<:Measurement}, N::Integer)
    return tensor_probability(rho, fill(Aax, N)...)
end
export tensor_probability

"""
    tensor_correlation(p::AbstractArray{T, N2}, behaviour::Bool = false; marg::Bool = true)

Converts a 2x...x2xmx...xm probability array into
- a mx...xm correlation array (no marginals)
- a (m+1)x...x(m+1) correlation array (marginals).
If `behaviour` is `true` do the transformation for behaviours. Doesn't assume normalization.

Also accepts the arguments of `tensor_probability` (state and measurements) for convenience.
"""
function tensor_correlation(p::AbstractArray{T, N2}, behaviour::Bool = false; marg::Bool = true) where {T, N2}
    @assert iseven(N2)
    N = N2 ÷ 2
    o = size(p)[1:N] # numbers of outputs per party
    @assert all(o .== 2)
    m = size(p)[(N + 1):end] # numbers of inputs per party
    size_FC = marg ? m .+ 1 : m
    FC = zeros(T, size_FC)
    cia = CartesianIndices(o)
    cix = CartesianIndices(size_FC)
    for x in cix
        # separating here prevent the need of the iterate function on unique elements of type T
        if all(x.I .> marg)
            FC[x] = sum((-1)^sum(a[n] - 1 for n in 1:N if x[n] > marg; init = 0) * p[a, (x.I .- marg)...] for a in cia)
        else
            x_colon = Union{Colon, Int64}[x[n] > marg ? x[n] - marg : Colon() for n in 1:N]
            FC[x] = sum((-1)^sum(a[n] - 1 for n in 1:N if x[n] > marg; init = 0) * sum(p[a, x_colon...]) for a in cia)
        end
    end
    if ~behaviour
        FC ./= 2^N
    elseif marg
        for n in 1:N
            x_colon = Union{Colon, Int64}[i == n ? 1 : Colon() for i in 1:N]
            FC[x_colon...] ./= m[n]
        end
    end
    cleanup!(FC)
    return FC
end
# accepts directly the arguments of tensor_probability
# avoids creating the full probability tensor for performance
function tensor_correlation(
        rho::Hermitian{T1, Matrix{T1}},
        first_Aax::Vector{Measurement{T2}}, # needed so that T2 is not unbounded
        other_Aax::Vector{Measurement{T2}}...;
        marg::Bool = true,
    ) where {T1, T2}
    T = real(promote_type(T1, T2))
    all_Aax = (first_Aax, other_Aax...)
    N = length(all_Aax)
    m = Tuple(length.(all_Aax)) # numbers of inputs per party
    o = Tuple(broadcast(Aax -> maximum(length.(Aax)), all_Aax)) # numbers of outputs per party
    @assert all(o .== 2)
    @assert all(broadcast(Aax -> minimum(length.(Aax)), all_Aax) .== 2) # sanity check
    size_FC = marg ? m .+ 1 : m
    FC = zeros(T, size_FC)
    cia = CartesianIndices(o)
    cix = CartesianIndices(size_FC)
    for a in cia, x in cix
        obs = [x[n] > marg ? all_Aax[n][x[n] - marg][1] - all_Aax[n][x[n] - marg][2] : one(all_Aax[n][1][1]) for n in 1:N]
        FC[x] = real(dot(Hermitian(kron(obs...)), rho))
    end
    return FC
end
# shorthand syntax for identical measurements on all parties
function tensor_correlation(rho::Hermitian, Aax::Vector{<:Measurement}, N::Integer; marg::Bool = true)
    return tensor_correlation(rho, fill(Aax, N)...; marg)
end
export tensor_correlation
