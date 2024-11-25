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

    for a in 0:(d - 1), b in 0:(d - 1), x in 0:(d - 1), y in 0:(d - 1)
        if mod(a + b + x * y, d) == 0
            G[a + 1, b + 1, x + 1, y + 1] = element
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
function cglmp(::Type{T}, d::Integer = 3) where {T}
    G = zeros(T, d, d, 2, 2)

    if T <: Integer
        normalization = 1
    else
        normalization = inv(T(4 * (d - 1)))
    end

    for a in 0:(d - 1), b in 0:(d - 1), x in 0:1, y in 0:1, k in 0:(d - 2)
        if mod(a - b, d) == mod((-1)^mod(x + y, 2) * k + x * y, d)
            G[a + 1, b + 1, x + 1, y + 1] = normalization * (d - 1 - k)
        end
    end

    return G
end
cglmp(d::Integer = 3) = cglmp(Float64, d)
export cglmp

"""
    inn22([T=Float64,] n::Integer = 3)

inn22 Bell functional in Collins-Gisin notation. Local bound 1.

Reference: Śliwa, [arXiv:quant-ph/0305190](https://arxiv.org/abs/quant-ph/0305190)
"""
function inn22(::Type{T}, n) where {T}
    C = zeros(T, n + 1, n + 1)
    for x in 1:n
        for y in 1:n
            if x + y <= n + 1
                C[x + 1, y + 1] = -1
            elseif x + y == n + 2
                C[x + 1, y + 1] = 1
            end
        end
    end
    C[1, 2] = 1
    C[2, 1] = 1
    return C
end
inn22(n::Integer = 3) = inn22(Int, n)
export inn22
