"""
    chsh([T=Float64,] d::Integer = 2)

CHSH-d nonlocal game in full probability notation. If `T` is an integer type the game is unnormalized.

Reference: Buhrman and Massar, [arXiv:quant-ph/0409066](https://arxiv.org/abs/quant-ph/0409066).
"""
function chsh(::Type{T}, d::Integer = 2) where {T}
    G = zeros(T, d, d, d, d)

    if T <: Integer
        element = 1
    else
        element = inv(T(d^2))
    end

    for a = 0:d-1, b = 0:d-1, x = 0:d-1, y = 0:d-1
        if mod(a + b + x * y, d) == 0
            G[a+1, b+1, x+1, y+1] = element
        end
    end

    return G
end
chsh(d::Integer = 2) = chsh(Float64, d)
export chsh

"""
    cglmp([T=Float64,] d::Integer)

CGLMP nonlocal game in full probability notation. If `T` is an integer type the game is unnormalized.

References: [arXiv:quant-ph/0106024](https://arxiv.org/abs/quant-ph/0106024) for the original game, and [arXiv:2005.13418](https://arxiv.org/abs/2005.13418) for the form presented here.
"""
function cglmp(::Type{T}, d::Integer) where {T}
    G = zeros(T, d, d, 2, 2)

    if T <: Integer
        normalization = 1
    else
        normalization = inv(T(4 * (d - 1)))
    end

    for a = 0:d-1, b = 0:d-1, x = 0:1, y = 0:1, k = 0:d-2
        if mod(a - b, d) == mod((-1)^mod(x + y, 2) * k + x * y, d)
            G[a+1, b+1, x+1, y+1] = normalization * (d - 1 - k)
        end
    end

    return G
end
cglmp(d::Integer) = cglmp(Float64, d)
export cglmp

# SD: not sure these functions belong here
"""
    probability_tensor(Aax::Vector{POVM{T}})

Applies N sets of POVMs onto a state `rho` to form a probability array.
"""
function probability_tensor(
    rho::LA.Hermitian{T1, Matrix{T1}},
    all_Aax::Vararg{Vector{POVM{T2}}, N},
) where {T1<:Number, T2<:Number, N}
    T = real(promote_type(T1, T2))
    m = length.(all_Aax) # numbers of inputs per party
    o = broadcast(Aax -> maximum(length.(Aax)), all_Aax) # numbers of outputs per party
    p = zeros(T, o..., m...)
    cia = CartesianIndices(o)
    cix = CartesianIndices(m)
    for a in cia, x in cix
        if all([a[n] ≤ length(all_Aax[n][x[n]]) for n in 1:N])
            p[a, x] = real(LA.tr(kron([all_Aax[n][x[n]][a[n]] for n in 1:N]...) * rho))
        end
    end
    return p
end
# accepts a pure state
function probability_tensor(
    psi::AbstractVector,
    all_Aax::Vararg{Vector{POVM{T}}, N},
) where {T<:Number, N}
    return probability_tensor(ketbra(psi), all_Aax...)
end
# accepts projective measurements
function probability_tensor(
    rho::LA.Hermitian{T1, Matrix{T1}},
    all_φax::Vararg{Vector{<:AbstractMatrix{T2}}, N},
) where {T1<:Number, T2<:Number, N}
    return probability_tensor(rho, povm.(all_φax)...)
end
# accepts pure states and projective measurements
function probability_tensor(
    psi::AbstractVector,
    all_φax::Vararg{Vector{<:AbstractMatrix{T}}, N},
) where {T<:Number, N}
    return probability_tensor(ketbra(psi), povm.(all_φax)...)
end
export probability_tensor

"""
    correlation_tensor(p::AbstractArray{T, N2}; marg::Bool = true)

Applies N sets of POVMs onto a state `rho` to form a probability array.
Convert a 2x...x2xmx...xm probability array into
- a mx...xm correlation array (no marginals)
- a (m+1)x...x(m+1) correlation array (marginals).
"""
function correlation_tensor(p::AbstractArray{T, N2}; marg::Bool = true) where {T<:Number} where {N2}
    @assert iseven(N2)
    N = N2 ÷ 2
    m = size(p)[N+1:end] # numbers of inputs per party
    o = size(p)[1:N] # numbers of outputs per party
    @assert collect(o) == 2ones(Int, N)
    res = zeros(T, (marg ? m .+ 1 : m)...)
    cia = CartesianIndices(Tuple(2ones(Int, N)))
    cix = CartesianIndices(Tuple(marg ? m .+ 1 : m))
    for x in cix
        x_colon = [x[n] ≤ m[n] ? x[n] : Colon() for n in 1:N]
        res[x] =
            sum((-1)^sum(a[n] for n in 1:N if x[n] ≤ m[n]; init = 0) * sum(p[a, x_colon...]) for a in cia) /
            prod(m[n] for n in 1:N if x[n] > m[n]; init = 1)
        if abs2(res[x]) < _tol(T)
            res[x] = 0
        end
    end
    return res
end
# accepts directly the arguments of probability_tensor
# SD: I'm still unsure whether it would be better practice to have a general syntax for this kind of argument passing
function correlation_tensor(
    rho::LA.Hermitian{T1, Matrix{T1}},
    all_Aax::Vararg{Vector{POVM{T2}}, N};
    marg::Bool = true,
) where {T1<:Number, T2<:Number, N}
    return correlation_tensor(probability_tensor(rho, all_Aax...); marg)
end
function correlation_tensor(
    psi::AbstractVector,
    all_Aax::Vararg{Vector{POVM{T}}, N};
    marg::Bool = true,
) where {T<:Number, N}
    return correlation_tensor(probability_tensor(psi, all_Aax...); marg)
end
function correlation_tensor(
    rho::LA.Hermitian{T1, Matrix{T1}},
    all_φax::Vararg{Vector{<:AbstractMatrix{T2}}, N},
) where {T1<:Number, T2<:Number, N}
    return correlation_tensor(probability_tensor(rho, all_φax))
end
function correlation_tensor(
    psi::AbstractVector,
    all_φax::Vararg{Vector{<:AbstractMatrix{T}}, N},
) where {T<:Number, N}
    return correlation_tensor(probability_tensor(psi, all_φax))
end
export correlation_tensor
