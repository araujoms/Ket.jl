"""
    white_noise(rho::AbstractMatrix, v::Real)

Returns `v * rho + (1 - v) * id`, where `id` is the maximally mixed state.
"""
function white_noise(rho::AbstractMatrix, v::Real)
    return white_noise!(copy(rho), v)
end
export white_noise

"""
    white_noise!(rho::AbstractMatrix, v::Real)

Modifies `rho` in place to tranform in into `v * rho + (1 - v) * id`
where `id` is the maximally mixed state.
"""
function white_noise!(rho::AbstractMatrix, v::Real)
    v == 1 && return rho
    parent(rho) .*= v
    tmp = (1 - v) / size(rho, 1)
    # https://discourse.julialang.org/t/change-the-diagonal-of-an-abstractmatrix-in-place/67294/2
    for i in axes(rho, 1)
        @inbounds rho[i, i] += tmp
    end
    return rho
end
export white_noise!

"""
    state_phiplus_ket([T=ComplexF64,] d::Integer = 2)

Produces the vector of the maximally entangled state Φ⁺ of local dimension `d`.
"""
function state_phiplus_ket(::Type{T}, d::Integer = 2; kwargs...) where {T<:Number}
    return state_ghz_ket(T, d, 2; kwargs...)
end
state_phiplus_ket(d::Integer = 2; kwargs...) = state_phiplus_ket(ComplexF64, d; kwargs...)
export state_phiplus_ket

"""
    state_phiplus([T=ComplexF64,] d::Integer = 2; v::Real = 1)

Produces the maximally entangled state Φ⁺ of local dimension `d` with visibility `v`.
"""
function state_phiplus(::Type{T}, d::Integer = 2; v::Real = 1) where {T<:Number}
    rho = ketbra(state_phiplus_ket(T, d; coeff = one(T)))
    parent(rho) ./= d
    return white_noise!(rho, v)
end
state_phiplus(d::Integer = 2; v::Real = 1) = state_phiplus(ComplexF64, d; v)
export state_phiplus

"""
    state_psiminus_ket([T=ComplexF64,] d::Integer = 2)

Produces the vector of the maximally entangled state ψ⁻ of local dimension `d`.
"""
function state_psiminus_ket(::Type{T}, d::Integer = 2; coeff = inv(_sqrt(T, d))) where {T<:Number}
    psi = zeros(T, d^2)
    psi[d.+(d-1)*(0:d-1)] .= (-1) .^ (0:d-1) .* coeff
    return psi
end
state_psiminus_ket(d::Integer = 2; kwargs...) = state_psiminus_ket(ComplexF64, d; kwargs...)
export state_psiminus_ket

"""
    state_psiminus([T=ComplexF64,] d::Integer = 2; v::Real = 1)

Produces the maximally entangled state ψ⁻ of local dimension `d` with visibility `v`.
"""
function state_psiminus(::Type{T}, d::Integer = 2; v::Real = 1) where {T<:Number}
    rho = ketbra(state_psiminus_ket(T, d; coeff = one(T)))
    parent(rho) ./= d
    return white_noise!(rho, v)
end
state_psiminus(d::Integer = 2; v::Real = 1) = state_psiminus(ComplexF64, d; v)
export state_psiminus

"""
    state_ghz_ket([T=ComplexF64,] d::Integer = 2, N::Integer = 3; coeff = 1/√d)

Produces the vector of the GHZ state local dimension `d`.
"""
function state_ghz_ket(::Type{T}, d::Integer = 2, N::Integer = 3; coeff = inv(_sqrt(T, d))) where {T<:Number}
    psi = zeros(T, d^N)
    spacing = (1 - d^N) ÷ (1 - d)
    psi[1:spacing:d^N] .= coeff
    return psi
end
state_ghz_ket(d::Integer = 2, N::Integer = 3; kwargs...) = state_ghz_ket(ComplexF64, d, N; kwargs...)
export state_ghz_ket

"""
    state_ghz([T=ComplexF64,] d::Integer = 2, N::Integer = 3; v::Real = 1, coeff = 1/√d)

Produces the GHZ state of local dimension `d` with visibility `v`.
"""
function state_ghz(::Type{T}, d::Integer = 2, N::Integer = 3; v::Real = 1, kwargs...) where {T<:Number}
    return white_noise!(ketbra(state_ghz_ket(T, d, N; kwargs...)), v)
end
state_ghz(d::Integer = 2, N::Integer = 3; kwargs...) = state_ghz(ComplexF64, d, N; kwargs...)
export state_ghz

"""
    state_w_ket([T=ComplexF64,] N::Integer = 3; coeff = 1/√d)

Produces the vector of the `N`-partite W state.
"""
function state_w_ket(::Type{T}, N::Integer = 3; coeff = inv(_sqrt(T, N))) where {T<:Number}
    psi = zeros(T, 2^N)
    psi[2 .^ (0:N-1).+1] .= coeff
    return psi
end
state_w_ket(N::Integer = 3; kwargs...) = state_w_ket(ComplexF64, N; kwargs...)
export state_w_ket

"""
    state_w([T=ComplexF64,] N::Integer = 3; v::Real = 1, coeff = 1/√d)

Produces the `N`-partite W state with visibility `v`.
"""
function state_w(::Type{T}, N::Integer = 3; v::Real = 1, kwargs...) where {T<:Number}
    return white_noise!(ketbra(state_w_ket(T, N; kwargs...)), v)
end
state_w(N::Integer = 3; kwargs...) = state_w(ComplexF64, N; kwargs...)
export state_w

"""
    isotropic(v::Real, d::Integer = 2)

Produces the isotropic state of local dimension `d` with visibility `v`.
"""
function isotropic(v::T, d::Integer = 2) where {T<:Real}
    return state_phiplus(T, d; v)
end
export isotropic

"""
    state_super_singlet_ket([T=ComplexF64,] N::Integer = 3; coeff = 1/√d)

Produces the vector of the `N`-partite `N`-level singlet state with visibility `v` (arXiv:quant-ph/0203119).
"""
function state_super_singlet_ket(::Type{T}, N::Integer = 3; coeff = inv(_sqrt(T, factorial(N)))) where {T<:Number}
    psi = zeros(T, N^N)
    for per in Combinatorics.permutations(1:N)
        tmp = kron(ket.((T,), per, (N,))...)
        if Combinatorics.parity(per) == 0
            # SD: this syntax allocates quite a bit
            # SD: we could go for an explicit one like for GHZ
            psi .+= tmp
        else
            # SD: this creates -0.0 in the output
            psi .-= tmp
        end
    end
    psi .*= coeff
    return psi
end
state_super_singlet_ket(N::Integer = 3; kwargs...) = state_super_singlet_ket(ComplexF64, N; kwargs...)
export state_super_singlet_ket

"""
    state_super_singlet([T=ComplexF64,] N::Integer = 3; v::Real = 1, coeff = 1/√d)

Produces the `N`-partite `N`-level singlet state with visibility `v` (arXiv:quant-ph/0203119).
"""
function state_super_singlet(::Type{T}, N::Integer = 3; v::Real = 1) where {T<:Number}
    rho = ketbra(state_super_singlet_ket(T, N; coeff = one(T)))
    parent(rho) ./= factorial(N)
    return white_noise!(rho, v)
end
state_super_singlet(N::Integer = 3; kwargs...) = state_super_singlet(ComplexF64, N; kwargs...)
export state_super_singlet

