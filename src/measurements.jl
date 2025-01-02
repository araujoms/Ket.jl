_root_unity(::Type{T}, n::Integer) where {T<:Number} = exp(2 * im * real(T)(π) / n)
_sqrt(::Type{T}, n::Integer) where {T<:Number} = sqrt(real(T)(n))

# MUBs
function mub_prime(::Type{T}, p::Integer) where {T<:Number}
    γ = _root_unity(T, p)
    inv_sqrt_p = inv(_sqrt(T, p))
    B = [Matrix{T}(undef, p, p) for _ ∈ 1:(p+1)]
    B[1] .= I(p)
    if p == 2
        B[2] .= [1 1; 1 -1] .* inv_sqrt_p
        B[3] .= [1 1; im -im] .* inv_sqrt_p
    else
        for k ∈ 0:(p-1)
            fill!(B[k+2], inv_sqrt_p)
            for t ∈ 0:(p-1), j ∈ 0:(p-1)
                exponent = mod(j * (t + k * j), p)
                if exponent == 0
                    continue
                elseif 4exponent == p
                    B[k+2][j+1, t+1] *= im
                elseif 2exponent == p
                    B[k+2][j+1, t+1] *= -1
                elseif 4exponent == 3p
                    B[k+2][j+1, t+1] *= -im
                else
                    B[k+2][j+1, t+1] *= γ^exponent
                end
            end
        end
    end
    return B
end
mub_prime(p::Integer) = mub_prime(ComplexF64, p)

function mub_prime_power(::Type{T}, p::Integer, r::Integer) where {T<:Number}
    d = Int64(p^r)
    γ = _root_unity(T, p)
    inv_sqrt_d = inv(_sqrt(T, d))
    B = [zeros(T, d, d) for _ ∈ 1:(d+1)]
    B[1] .= I(d)
    f, x = Nemo.finite_field(p, r, "x")
    pow = [x^i for i ∈ 0:(r-1)]
    el = [sum(digits(i; base = p, pad = r) .* pow) for i ∈ 0:(d-1)]
    if p == 2
        for i ∈ 1:d, k ∈ 0:(d-1), q ∈ 0:(d-1)
            aux = one(T)
            q_bin = digits(q; base = 2, pad = r)
            for m ∈ 0:(r-1), n ∈ 0:(r-1)
                aux *= conj(im^_tr_ff(el[i] * el[q_bin[m+1]*2^m+1] * el[q_bin[n+1]*2^n+1]))
            end
            B[i+1][:, k+1] += (-1)^_tr_ff(el[q+1] * el[k+1]) * aux * B[1][:, q+1] * inv_sqrt_d
        end
    else
        inv_two = inv(2 * one(f))
        for i ∈ 1:d, k ∈ 0:(d-1), q ∈ 0:(d-1)
            B[i+1][:, k+1] +=
                γ^_tr_ff(-el[q+1] * el[k+1]) * γ^_tr_ff(el[i] * el[q+1] * el[q+1] * inv_two) * B[1][:, q+1] * inv_sqrt_d
        end
    end
    return B
end
mub_prime_power(p::Integer, r::Integer) = mub_prime_power(ComplexF64, p, r)

# auxiliary function to compute the trace in finite fields as an Int64
function _tr_ff(a::Nemo.FqFieldElem)
    Int64(Nemo.lift(Nemo.ZZ, Nemo.absolute_tr(a)))
end

"""
    mub([T=ComplexF64,] d::Integer)

Construction of the standard complete set of MUBs.
The output contains 1+minᵢ pᵢ^rᵢ bases, where `d` = ∏ᵢ pᵢ^rᵢ.

Reference: Durt, Englert, Bengtsson, Życzkowski, [arXiv:1004.3348](https://arxiv.org/abs/1004.3348).
"""
function mub(::Type{T}, d::Integer) where {T<:Number}
    # the dimension d can be any integer greater than two
    @assert d ≥ 2
    f = collect(Nemo.factor(Int64(d))) # Nemo.factor requires d to be an Int64 (or UInt64)
    p = f[1][1]
    r = f[1][2]
    if length(f) > 1 # different prime factors
        B_aux1 = mub(T, p^r)
        B_aux2 = mub(T, d ÷ p^r)
        k = min(length(B_aux1), length(B_aux2))
        B = [Matrix{T}(undef, d, d) for _ ∈ 1:k]
        for j ∈ 1:k
            B[j] .= kron(B_aux1[j], B_aux2[j])
        end
    elseif r == 1 # prime
        return mub_prime(T, p)
    else # prime power
        return mub_prime_power(T, p, r)
    end
    return B
end
mub(d::Integer) = mub(ComplexF64, d)
export mub

# Select a specific subset with k bases
function mub(::Type{T}, d::Integer, k::Integer, s::Integer = 1) where {T<:Number}
    B = mub(T, d)
    subs = collect(Iterators.take(Combinatorics.combinations(1:length(B), k), s))
    sub = subs[end]
    return B[sub]
end
mub(d::Integer, k::Integer, s::Integer = 1) = mub(ComplexF64, d, k, s)

"""
    test_mub(B::Vector{Matrix{<:Number}})

Checks if the input bases are mutually unbiased.
"""
function test_mub(B::Vector{Matrix{T}}) where {T<:Number}
    d = size(B[1], 1)
    k = length(B)
    inv_d = inv(T(d))
    for x ∈ 1:k, y ∈ x:k, a ∈ 1:d, b ∈ 1:d
        # expected scalar product squared
        if x == y
            sc2_exp = T(a == b)
        else
            sc2_exp = inv_d
        end
        sc2 = abs2(dot(B[x][:, a], B[y][:, b]))
        if abs2(sc2 - sc2_exp) > _eps(T)
            return false
        end
    end
    return true
end
export test_mub

# SIC POVMs

"""
    sic_povm([T=ComplexF64,] d::Integer)

Constructs a vector of `d²` vectors |vᵢ⟩ such that |vᵢ⟩⟨vᵢ| forms a SIC-POVM of dimension `d`. This construction is based on the Weyl-Heisenberg fiducial.

Reference: Appleby, Yadsan-Appleby, Zauner, [arXiv:1209.1813](http://arxiv.org/abs/1209.1813)
"""
function sic_povm(::Type{T}, d::Integer) where {T}
    fiducial = _fiducial_WH(real(T), d)
    vecs = Vector{Vector{T}}(undef, d^2)
    for p ∈ 0:(d-1)
        Xp = shift(T, d, p)
        for q ∈ 0:(d-1)
            Zq = clock(T, d, q)
            vecs[d*p+q+1] = Xp * Zq * fiducial
        end
    end
    sqrt_d = _sqrt(T, d)
    for vi ∈ vecs
        vi ./= sqrt_d * norm(vi)
    end
    return vecs
end
sic_povm(d::Integer) = sic_povm(ComplexF64, d)
export sic_povm

"""
    test_sic(vecs)

Checks if `vecs` is a vector of `d²` vectors |vᵢ⟩ such that |vᵢ⟩⟨vᵢ| forms a SIC-POVM of dimension `d`.
"""
function test_sic(vecs::Vector{Vector{T}}) where {T<:Number}
    d = length(vecs[1])
    length(vecs) == d^2 || throw(ArgumentError("Number of vectors must be d² = $(d^2), got $(length(vecs))."))
    normalization = inv(T(d^2))
    symmetry = inv(T(d^2 * (d + 1)))
    for j ∈ 1:(d^2), i ∈ 1:j
        inner_product = abs2(dot(vecs[i], vecs[j]))
        if i == j
            deviation = abs2(inner_product - normalization)
        else
            deviation = abs2(inner_product - symmetry)
        end
        if deviation > _eps(T)
            return false
        end
    end
    return true
end
export test_sic

"""
    dilate_povm(vecs::Vector{Vector{T}})

Does the Naimark dilation of a rank-1 POVM given as a vector of vectors. This is the minimal dilation.
"""
function dilate_povm(vecs::Vector{Vector{T}}) where {T<:Number}
    d = length(vecs[1])
    n = length(vecs)
    V = Matrix{T}(undef, n, d)
    for j ∈ 1:d
        for i ∈ 1:n
            V[i, j] = conj(vecs[i][j])
        end
    end
    return V
end
export dilate_povm

"""
    dilate_povm(E::Vector{<:AbstractMatrix})

Does the Naimark dilation of a POVM given as a vector of matrices. This always works, but is wasteful if the POVM elements are not full rank.
"""
function dilate_povm(E::Vector{<:AbstractMatrix})
    n = length(E)
    rtE = sqrt.(E)
    return sum(kron(rtE[i], ket(i, n)) for i ∈ 1:n)
end

"""
    _fiducial_WH([T=ComplexF64,] d::Integer)

Computes the fiducial Weyl-Heisenberg vector of dimension `d`.

Reference: Appleby, Yadsan-Appleby, Zauner, [arXiv:1209.1813](http://arxiv.org/abs/1209.1813) http://www.gerhardzauner.at/sicfiducials.html
"""
function _fiducial_WH(::Type{T}, d::Integer) where {T}
    if d == 1
        return [T(1)]
    elseif d == 2
        return [sqrt(0.5 * (1 + 1 / sqrt(T(3)))), exp(im * T(π) / 4) * sqrt(0.5 * (1 - 1 / sqrt(T(3))))]
    elseif d == 3
        return [T(0), T(1), T(1)]
    elseif d == 4
        a = sqrt(T(5))
        r = sqrt(T(2))
        b = im * sqrt(a + 1)
        return [
            T(1) / 40 * (-a + 5) * r + T(1) / 20 * (-a + 5),
            ((-T(1) / 40 * a * r - T(1) / 40 * a) * b + (T(1) / 80 * (a - 5) * r + T(1) / 40 * (a - 5))) * im +
            -T(1) / 40 * a * b +
            T(1) / 80 * (-a + 5) * r,
            T(1) / 40 * (a - 5) * r * im,
            ((T(1) / 40 * a * r + T(1) / 40 * a) * b + (T(1) / 80 * (a - 5) * r + T(1) / 40 * (a - 5))) * im +
            T(1) / 40 * a * b +
            T(1) / 80 * (-a + 5) * r
        ]
    elseif d == 5
        a = sqrt(T(3))
        r = sqrt(T(5))
        t = sin(T(π) / 5)
        b = im * sqrt(5a + (5 + 3r) * t)
        return [
            T(1) / 60 * (-a + 3) * r + T(1) / 12 * (-a + 3),
            (
                ((T(1) / 120 * a * r + T(1) / 40 * a) * t + (-T(1) / 240 * a * r + T(1) / 240 * a)) * b +
                (T(1) / 120 * (a - 3) * r * t + (T(1) / 32 * (-a + 1) * r + T(1) / 32 * (-a + 1)))
            ) * im +
            ((-T(1) / 120 * a * r - T(1) / 120 * a) * t - T(1) / 120 * a) * b +
            T(1) / 40 * (a - 1) * r * t +
            T(1) / 160 * (-a + 3) * r +
            T(1) / 96 * (a - 3),
            (
                (
                    (-T(1) / 80 * a * r + T(1) / 240 * (-7 * a + 6)) * t +
                    (T(1) / 480 * (-a + 9) * r + T(1) / 480 * (-a + 15))
                ) * b + (T(1) / 120 * (-a + 3) * r * t + (T(1) / 32 * (-a + 1) * r + T(1) / 32 * (-a + 1)))
            ) * im +
            (
                (T(1) / 240 * (2 * a + 3) * r + T(1) / 240 * (4 * a + 3)) * t +
                (T(1) / 480 * (-a - 3) * r + T(1) / 160 * (-a - 5))
            ) * b +
            T(1) / 40 * (-a + 1) * r * t +
            T(1) / 160 * (-a + 3) * r +
            T(1) / 96 * (a - 3),
            (
                (
                    (T(1) / 80 * a * r + T(1) / 240 * (7 * a - 6)) * t +
                    (T(1) / 480 * (a - 9) * r + T(1) / 480 * (a - 15))
                ) * b + (T(1) / 120 * (-a + 3) * r * t + (T(1) / 32 * (-a + 1) * r + T(1) / 32 * (-a + 1)))
            ) * im +
            (
                (T(1) / 240 * (-2 * a - 3) * r + T(1) / 240 * (-4 * a - 3)) * t +
                (T(1) / 480 * (a + 3) * r + T(1) / 160 * (a + 5))
            ) * b +
            T(1) / 40 * (-a + 1) * r * t +
            T(1) / 160 * (-a + 3) * r +
            T(1) / 96 * (a - 3),
            (
                ((-T(1) / 120 * a * r - T(1) / 40 * a) * t + (T(1) / 240 * a * r - T(1) / 240 * a)) * b +
                (T(1) / 120 * (a - 3) * r * t + (T(1) / 32 * (-a + 1) * r + T(1) / 32 * (-a + 1)))
            ) * im +
            ((T(1) / 120 * a * r + T(1) / 120 * a) * t + T(1) / 120 * a) * b +
            T(1) / 40 * (a - 1) * r * t +
            T(1) / 160 * (-a + 3) * r +
            T(1) / 96 * (a - 3)
        ]
    elseif d == 6
        a = sqrt(T(21))
        r = [sqrt(T(2)), sqrt(T(3))]
        b = [im * sqrt(2a + 6), 2real((1 + im * sqrt(T(7)))^(1 // 3))]
        return [
            (
                T(1) / 1008 * (-a + 3) * r[2] * b[1] * b[2]^2 +
                T(1) / 504 * (a - 6) * r[2] * b[1] * b[2] +
                (T(1) / 168 * (a - 2) * r[2] - T(1) / 168 * a) * b[1]
            ) * im +
            T(1) / 504 * (-a + 5) * r[2] * b[2]^2 +
            T(1) / 504 * (a + 1) * r[2] * b[2] +
            T(1) / 504 * (-a - 13) * r[2] +
            T(1) / 168 * (a + 21),
            (
                (T(1) / 2016 * (-a + 3) * r[2] * b[1] + (T(1) / 504 * (a - 7) * r[2] + T(1) / 336 * (a - 5))) * b[2]^2 +
                (
                    (T(1) / 1008 * (a - 6) * r[2] + T(1) / 672 * (-a + 7)) * b[1] +
                    (T(1) / 1008 * (-a - 7) * r[2] + T(1) / 336 * (-a - 1))
                ) * b[2] +
                (
                    (T(1) / 672 * (a + 3) * r[2] + T(1) / 672 * (-3 * a + 7)) * b[1] +
                    (T(1) / 504 * (-3 * a + 14) * r[2] + T(1) / 168 * (-a - 4))
                )
            ) * im +
            (
                (T(1) / 1008 * (-a + 3) * r[2] + T(1) / 672 * (a - 3)) * b[1] +
                (T(1) / 1008 * (-3 * a + 7) * r[2] - T(1) / 84)
            ) * b[2]^2 +
            (
                (T(1) / 2016 * (a - 3) * r[2] - T(1) / 336) * b[1] +
                (T(1) / 1008 * (5 * a - 7) * r[2] + T(1) / 336 * (a - 5))
            ) * b[2] +
            (T(1) / 224 * (a - 5) * r[2] + T(1) / 672 * (-7 * a + 19)) * b[1] +
            T(1) / 72 * (a - 4) * r[2] +
            T(1) / 168 * (-a + 22),
            (
                (T(1) / 672 * (-a + 3) * r[2] * b[1] + T(1) / 336 * (a - 5)) * b[2]^2 +
                (T(1) / 336 * r[2] * b[1] + T(1) / 336 * (-a - 1)) * b[2] +
                (
                    (T(1) / 672 * (5 * a - 19) * r[2] + T(1) / 224 * (-a + 7)) * b[1] +
                    (T(1) / 336 * (5 * a - 21) * r[2] + T(1) / 336 * (-5 * a + 13))
                )
            ) * im +
            (T(1) / 672 * (-a + 3) * b[1] + T(1) / 1008 * (a - 5) * r[2]) * b[2]^2 +
            (T(1) / 336 * b[1] + T(1) / 1008 * (-a - 1) * r[2]) * b[2] +
            (T(1) / 672 * (-a + 7) * r[2] + T(1) / 672 * (5 * a - 19)) * b[1] +
            T(1) / 1008 * (-5 * a + 13) * r[2] +
            T(1) / 336 * (5 * a - 21),
            (
                (T(1) / 504 * (a - 3) * r[2] * b[1] + T(1) / 504 * (a - 1) * r[2]) * b[2]^2 +
                (T(1) / 1008 * (-a + 3) * r[2] * b[1] + T(1) / 252 * (-a + 2) * r[2]) * b[2] +
                (T(1) / 168 * (-a + 4) * r[2] * b[1] + T(1) / 504 * (-9 * a + 11) * r[2])
            ) * im +
            (T(1) / 1008 * (a - 3) * r[2] * b[1] - T(1) / 126 * r[2]) * b[2]^2 +
            (T(1) / 1008 * (a - 9) * r[2] * b[1] + T(1) / 504 * (a - 5) * r[2]) * b[2] +
            T(1) / 168 * (-a + 2) * r[2] * b[1] +
            T(1) / 504 * (-5 * a + 23) * r[2],
            (
                (T(1) / 2016 * (-a + 3) * r[2] * b[1] + T(1) / 336 * (-a + 1)) * b[2]^2 +
                (T(1) / 2016 * (-a + 9) * r[2] * b[1] + T(1) / 168 * (a - 2)) * b[2] +
                (
                    (T(1) / 672 * (a + 3) * r[2] + T(1) / 672 * (a - 21)) * b[1] +
                    (T(1) / 168 * a * r[2] + T(1) / 168 * (3 * a - 16))
                )
            ) * im +
            (T(1) / 672 * (-a + 3) * b[1] + T(1) / 1008 * (a + 7) * r[2]) * b[2]^2 +
            (T(1) / 672 * (a - 5) * b[1] + T(1) / 504 * (-2 * a + 7) * r[2]) * b[2] +
            (T(1) / 672 * (3 * a - 7) * r[2] + T(1) / 672 * (a - 5)) * b[1] +
            T(1) / 504 * (-a - 28) * r[2] +
            T(1) / 168 * a,
            (
                (T(1) / 672 * (a - 3) * b[1] + (T(1) / 1008 * (-a + 1) * r[2] + T(1) / 84)) * b[2]^2 +
                (
                    (T(1) / 672 * (a - 7) * r[2] + T(1) / 672 * (-a + 5)) * b[1] +
                    (T(1) / 504 * (a - 2) * r[2] + T(1) / 336 * (-a + 5))
                ) * b[2] +
                (
                    (T(1) / 672 * (a - 7) * r[2] + T(1) / 672 * (-3 * a + 5)) * b[1] +
                    (T(1) / 1008 * (3 * a - 11) * r[2] + T(1) / 336 * (-a - 23))
                )
            ) * im +
            (T(1) / 672 * (-a + 3) * r[2] * b[1] + (T(1) / 252 * r[2] + T(1) / 336 * (a - 1))) * b[2]^2 +
            (
                (T(1) / 672 * (a - 5) * r[2] + T(1) / 672 * (a - 7)) * b[1] +
                (T(1) / 1008 * (-a + 5) * r[2] + T(1) / 168 * (-a + 2))
            ) * b[2] +
            (T(1) / 672 * (3 * a - 5) * r[2] + T(1) / 672 * (a - 7)) * b[1] +
            T(1) / 1008 * (-a - 23) * r[2] +
            T(1) / 336 * (-3 * a + 11)
        ]
    elseif d == 7
        a = sqrt(T(2))
        r = sqrt(T(7))
        t = cos(T(π) / 7)
        b = im * sqrt(2a + 1)
        return [
            T(1) / 14 * (a + 1),
            (
                (T(1) / 196 * (a - 4) * r * t^2 + T(1) / 392 * (3 * a + 2) * r * t + T(1) / 392 * (a + 3) * r) * b +
                (T(1) / 196 * (a + 6) * r * t^2 + T(1) / 392 * (-3 * a - 4) * r * t + T(1) / 392 * (-5 * a - 9) * r)
            ) * im +
            (-T(1) / 28 * a * t^2 + T(1) / 56 * (a - 2) * t + T(1) / 56 * (a - 1)) * b +
            T(1) / 28 * (3 * a + 6) * t^2 +
            T(1) / 56 * (-a - 4) * t +
            T(1) / 56 * (-3 * a - 5),
            (
                (T(1) / 196 * (3 * a + 2) * r * t^2 + T(1) / 196 * (-2 * a + 1) * r * t + T(1) / 784 * (a - 4) * r) *
                b +
                (T(1) / 196 * (-3 * a - 4) * r * t^2 + T(1) / 196 * (a - 1) * r * t + T(1) / 784 * (-5 * a - 2) * r)
            ) * im +
            (T(1) / 28 * (a - 2) * t^2 + T(1) / 28 * t - T(1) / 112 * a) * b +
            T(1) / 28 * (-a - 4) * t^2 +
            T(1) / 28 * (-a - 1) * t +
            T(1) / 112 * (a + 6),
            (
                (T(1) / 98 * (2 * a - 1) * r * t^2 + T(1) / 392 * (-a + 4) * r * t + T(1) / 784 * (-11 * a + 2) * r) *
                b + (T(1) / 98 * (a - 1) * r * t^2 + T(1) / 392 * (a + 6) * r * t + T(1) / 784 * (-13 * a - 8) * r)
            ) * im +
            (-T(1) / 14 * t^2 + T(1) / 56 * a * t + T(1) / 112 * (-a + 6)) * b +
            T(1) / 14 * (-a - 1) * t^2 +
            T(1) / 56 * (3 * a + 6) * t +
            T(1) / 112 * a,
            (
                (T(1) / 98 * (-2 * a + 1) * r * t^2 + T(1) / 392 * (a - 4) * r * t + T(1) / 784 * (11 * a - 2) * r) *
                b + (T(1) / 98 * (a - 1) * r * t^2 + T(1) / 392 * (a + 6) * r * t + T(1) / 784 * (-13 * a - 8) * r)
            ) * im +
            (T(1) / 14 * t^2 - T(1) / 56 * a * t + T(1) / 112 * (a - 6)) * b +
            T(1) / 14 * (-a - 1) * t^2 +
            T(1) / 56 * (3 * a + 6) * t +
            T(1) / 112 * a,
            (
                (T(1) / 196 * (-3 * a - 2) * r * t^2 + T(1) / 196 * (2 * a - 1) * r * t + T(1) / 784 * (-a + 4) * r) *
                b +
                (T(1) / 196 * (-3 * a - 4) * r * t^2 + T(1) / 196 * (a - 1) * r * t + T(1) / 784 * (-5 * a - 2) * r)
            ) * im +
            (T(1) / 28 * (-a + 2) * t^2 - T(1) / 28 * t + T(1) / 112 * a) * b +
            T(1) / 28 * (-a - 4) * t^2 +
            T(1) / 28 * (-a - 1) * t +
            T(1) / 112 * (a + 6),
            (
                (T(1) / 196 * (-a + 4) * r * t^2 + T(1) / 392 * (-3 * a - 2) * r * t + T(1) / 392 * (-a - 3) * r) * b +
                (T(1) / 196 * (a + 6) * r * t^2 + T(1) / 392 * (-3 * a - 4) * r * t + T(1) / 392 * (-5 * a - 9) * r)
            ) * im +
            (T(1) / 28 * a * t^2 + T(1) / 56 * (-a + 2) * t + T(1) / 56 * (-a + 1)) * b +
            T(1) / 28 * (3 * a + 6) * t^2 +
            T(1) / 56 * (-a - 4) * t +
            T(1) / 56 * (-3 * a - 5)
        ]
    else
        throw(ArgumentError(string("Invalid input dimension d = ", d)))
    end
end
