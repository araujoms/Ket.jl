"""
    local_bound(G::Array{T,N})

Computes the local bound of a multipartite Bell functional `G`, written in full probability notation
as an `N`-dimensional array.

Reference: Araújo, Hirsch, and Quintino, [arXiv:2005.13418](https://arxiv.org/abs/2005.13418).
"""
function local_bound(G::Array{T, N2}) where {T <: Real, N2}
    @assert iseven(N2)
    N = N2 ÷ 2
    scenario = size(G)
    outs = scenario[1:N]
    ins = scenario[(N + 1):(2 * N)]

    num_strategies = outs .^ ins
    largest_party = argmax(num_strategies)
    if largest_party != 1
        perm = [largest_party; 2:(largest_party - 1); 1; (largest_party + 1):N]
        outs = outs[perm]
        ins = ins[perm]
        bigperm::NTuple{N2, Int} = Tuple([perm; perm .+ N])
        G = permutedims(G, bigperm)
    end

    perm::NTuple{N2, Int} = Tuple([1; N + 1; 2:N; (N + 2):(2 * N)])
    permutedG = permutedims(G, perm)
    squareG = reshape(permutedG, outs[1] * ins[1], prod(outs[2:N]) * prod(ins[2:N]))

    chunks = _partition(prod((outs .^ ins)[2:N]), Threads.nthreads())
    outs2 = outs; ins2 = ins; squareG2 = squareG  #workaround for https://github.com/JuliaLang/julia/issues/15276
    tasks = map(chunks) do chunk
        Threads.@spawn _local_bound_single(chunk, outs2, ins2, squareG2)
    end
    score = maximum(fetch.(tasks))

    return score
end
export local_bound

function _local_bound_single(chunk, outs::NTuple{2, Int}, ins::NTuple{2, Int}, squareG::Array{T, 2}) where {T}
    oa, ob = outs
    ia, ib = ins
    score = typemin(T)
    ind = digits(chunk[1] - 1; base = ob, pad = ib)
    offset = Vector(1 .+ ob * (0:(ib - 1)))
    offset_ind = zeros(Int, ib)
    Galice = zeros(T, oa * ia)
    @inbounds for _ in chunk[1]:chunk[2]
        offset_ind .= ind .+ offset
        @views sum!(Galice, squareG[:, offset_ind])
        temp_score = _maxcols!(Galice, oa, ia)
        score = max(score, temp_score)
        _update_odometer!(ind, ob)
    end

    return score
end

function _local_bound_single(chunk, outs::NTuple{N, Int}, ins::NTuple{N, Int}, squareG::Array{T, 2}) where {T, N}
    score = typemin(T)
    bases = reduce(vcat, [outs[i] * ones(Int, ins[i]) for i in 2:length(ins)])
    ind = _digits_mixed_basis(chunk[1] - 1, bases)
    Galice = zeros(T, outs[1] * ins[1])
    b = zeros(Int, N - 1)
    sizes = (outs[2:N]..., ins[2:N]...)
    prodsizes = ones(Int, 2 * (N - 1))
    for i in 1:length(prodsizes)
        prodsizes[i] = prod(sizes[1:(i - 1)])
    end
    linearindex(v) = 1 + dot(v, prodsizes)
    by = zeros(Int, 2 * (N - 1))
    @inbounds for _ in chunk[1]:chunk[2]
        fill!(Galice, 0)
        for y in CartesianIndices(ins[2:N])
            by[1] = ind[y[1]]
            for i in 2:length(y)
                by[i] = ind[y[i] + ins[i]]
            end
            for i in 1:(N - 1)
                by[i + N - 1] = y[i] - 1
            end
            for i in 1:outs[1]*ins[1]
                Galice[i] += squareG[i, linearindex(by)]
            end
        end
        temp_score = _maxcols!(Galice, outs[1], ins[1])
        score = max(score, temp_score)
        _update_odometer!(ind, bases)
    end

    return score
end

#sum(maximum(v, dims = 1)), with v interpreted as a oa x ia matrix
function _maxcols!(v, oa, ia)
    for x = 1:ia
        for a = 2:oa
            if v[a + (x-1)*oa] > v[1 + (x-1)*oa]
                v[1 + (x-1)*oa] = v[a + (x-1)*oa]
            end
        end
    end
    temp_score = v[1]
    for x = 2:ia
        temp_score += v[1 + (x-1)*oa]
    end
    return temp_score
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

function _digits_mixed_basis(ind, bases)
    N = length(bases)
    digits = zeros(Int, N)
    for i in N:-1:1
        digits[i] = mod(ind, bases[i])
        ind = div(ind, bases[i])
    end
    return digits
end

function _update_odometer!(ind::AbstractVector{<:Integer}, bases::AbstractVector{<:Integer})
    ind[1] += 1
    d = length(ind)

    for i in 1:d
        if ind[i] ≥ bases[i]
            ind[i] = 0
            i < d ? ind[i + 1] += 1 : return
        else
            return
        end
    end
end

function _update_odometer!(ind::AbstractVector{<:Integer}, bases::Integer)
    ind[1] += 1
    d = length(ind)

    for i in 1:d
        if ind[i] ≥ bases
            ind[i] = 0
            i < d ? ind[i + 1] += 1 : return
        else
            return
        end
    end
end

"""
    tensor_collinsgisin(p::Array, behaviour::Bool = false)

Takes a multipartite Bell functional `p` in full probability notation and transforms it to Collins-Gisin notation.
If `behaviour` is `true` do instead the transformation for behaviours. Doesn't assume normalization.

Also accepts the arguments of `tensor_probability` (state and measurements) for convenience.
"""
function tensor_collinsgisin(p::AbstractArray{T, N2}, behaviour::Bool = false) where {T, N2}
    @assert iseven(N2)
    N = N2 ÷ 2
    scenario = size(p)
    outs = scenario[1:N]
    ins = scenario[(N + 1):(2 * N)]
    cgindex(a, x) = (a .!= outs) .* (a .+ (x .- 1) .* (outs .- 1)) .+ 1
    CG = zeros(T, ins .* (outs .- 1) .+ 1)

    if ~behaviour
        for x in CartesianIndices(ins)
            for a in CartesianIndices(outs)
                for a2 in Iterators.product(union.(a.I, outs)...)
                    ndiff = abs(sum(a.I .!= outs) - sum(a2 .!= outs))
                    CG[cgindex(a.I, x.I)...] += (-1)^ndiff * p[a2..., x]
                end
            end
        end
    else
        for x in CartesianIndices(ins)
            for a in CartesianIndices(outs)
                cgiterators = map((i, j) -> i == j ? (1:j) : (i:i), a.I, outs)
                for a2 in CartesianIndices(cgiterators)
                    CG[cgindex(a.I, x.I)...] += p[a2, x] / prod(ins[BitVector(a.I .== outs)])
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
    tensor_probability(CG::Array, scenario::AbstractVecOrTuple, behaviour::Bool = false)

Takes a multipartite Bell functional `CG` in Collins-Gisin notation and transforms it to full probability notation.
`scenario` is a tuple detailing the number of inputs and outputs, in the order (oa, ob, ..., ia, ib, ...).
If `behaviour` is `true` do instead the transformation for behaviours. Doesn't assume normalization.
"""
function tensor_probability(CG::AbstractArray{T, N}, scenario::AbstractVecOrTuple{<:Integer}, behaviour::Bool = false) where {T, N}
    p = zeros(T, scenario...)
    outs = Tuple(scenario[1:N])
    ins = Tuple(scenario[(N + 1):(2 * N)])
    cgindex(a, x) = (a .!= outs) .* (a .+ (x .- 1) .* (outs .- 1)) .+ 1

    if ~behaviour
        for x in CartesianIndices(ins)
            for a in CartesianIndices(outs)
                for a2 in Iterators.product(union.(a.I, outs)...)
                    p[a, x] += CG[cgindex(a2, x.I)...] / prod(ins[BitVector(a2 .== outs)])
                end
            end
        end
    else
        for x in CartesianIndices(ins)
            for a in CartesianIndices(outs)
                cgiterators = map((i, j) -> i == j ? (1:j) : (i:i), a.I, outs)
                for a2 in CartesianIndices(cgiterators)
                    ndiff = abs(sum(a.I .!= outs) - sum(a2.I .!= outs))
                    p[a, x] += (-1)^ndiff * CG[cgindex(a2.I, x.I)...]
                end
            end
        end
    end
    return p
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
            x_colon = Union{Colon, Int}[x[n] > marg ? x[n] - marg : Colon() for n in 1:N]
            FC[x] = sum((-1)^sum(a[n] - 1 for n in 1:N if x[n] > marg; init = 0) * sum(p[a, x_colon...]) for a in cia)
        end
    end
    if ~behaviour
        FC ./= 2^N
    elseif marg
        for n in 1:N
            x_colon = Union{Colon, Int}[i == n ? 1 : Colon() for i in 1:N]
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
