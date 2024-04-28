"""
    chsh(d::Integer = 2)

CHSH-d nonlocal game in full probability notation

Reference: Buhrman and Massar, https://arxiv.org/abs/quant-ph/0409066.
"""
function chsh(d::Integer = 2; T::Type = Float64)
    G = zeros(T, d, d, d, d)

    for a = 0:d-1, b = 0:d-1, x = 0:d-1, y = 0:d-1
        if mod(a + b + x * y, d) == 0
            G[a+1, b+1, x+1, y+1] = 1
        end
    end

    #G /= d^2
    return G
end
export chsh

"""
    cglmp(d::Integer)

CGLMP nonlocal game in full probability notation

References: https://arxiv.org/abs/quant-ph/0106024 for the original game, and
https://arxiv.org/abs/2005.13418 for the form presented here.
"""
function cglmp(d::Integer; T::Type = Float64)
    V = zeros(T, d, d, 2, 2)

    for a = 0:d-1, b = 0:d-1, x = 0:1, y = 0:1
        for k = 0:d-2
            V[a+1, b+1, x+1, y+1] += (1 - T(k) / (d - 1)) * (mod(a - b, d) == mod((-1)^mod(x + y, 2) * k + x * y, d))
        end
    end
    V /= 4

    return V
end
export cglmp
