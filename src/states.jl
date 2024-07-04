"""
    white_noise(rho::LA.Hermitian, v::Real)

Returns `v * rho + (1 - v) * id`, where `id` is the maximally mixed state.
"""
function white_noise(rho::AbstractMatrix, v::Real)
    return white_noise!(copy(rho), v)
end
export white_noise

"""
    white_noise!(rho::LA.Hermitian, v::Real)

Modifies `rho` in place to tranform in into `v * rho + (1 - v) * id`
where `id` is the maximally mixed state.
"""
function white_noise!(rho::AbstractMatrix, v::Real)
    if v != 1
        rho .*= v
        tmp = (1 - v) / size(rho, 1)
        # https://discourse.julialang.org/t/change-the-diagonal-of-an-abstractmatrix-in-place/67294/2
        for i in axes(rho, 1)
            @inbounds rho[i, i] += tmp
        end
    end
    return rho
end
export white_noise!

"""
    state_phiplus_ket([T=ComplexF64,] d::Integer = 2)

Produces the vector of the maximally entangled state Φ⁺ of local dimension `d`.
"""
function state_phiplus_ket(::Type{T}, d::Integer = 2) where {T<:Number}
    psi = zeros(T, d^2)
    for i = 1:d
        psi += kron(ket(T, i, d), ket(T, i, d))
    end
    return psi
end
state_phiplus_ket(d::Integer = 2) = state_phiplus_ket(ComplexF64, d)
export state_phiplus_ket

"""
    state_phiplus([T=ComplexF64,] d::Integer = 2; v::Real = 1)

Produces the maximally entangled state Φ⁺ of local dimension `d` with visibility `v`.
"""
function state_phiplus(::Type{T}, d::Integer = 2; v::Real = 1) where {T<:Number}
    return white_noise!(ketbra(state_phiplus_ket(T, d)), v)
end
state_phiplus(d::Integer = 2; v::Real = 1) = state_phiplus(ComplexF64, d; v)
export state_phiplus

"""
    state_psiminus_ket([T=ComplexF64,] d::Integer = 2)

Produces the vector of the maximally entangled state ψ⁻ of local dimension `d`.
"""
function state_psiminus_ket(::Type{T}, d::Integer = 2) where {T<:Number}
    psi = zeros(T, d^2)
    for i = 1:d
        psi += (-1)^(i + 1) * kron(ket(T, i, d), ket(T, d - i + 1, d))
    end
    return psi
end
state_psiminus_ket(d::Integer = 2) = state_psiminus_ket(ComplexF64, d)
export state_psiminus_ket

"""
    state_psiminus([T=ComplexF64,] d::Integer = 2; v::Real = 1)

Produces the maximally entangled state ψ⁻ of local dimension `d` with visibility `v`.
"""
function state_psiminus(::Type{T}, d::Integer = 2; v::Real = 1) where {T<:Number}
    return white_noise!(ketbra(state_psiminus_ket(T, d)), v)
end
state_psiminus(d::Integer = 2; v::Real = 1) = state_psiminus(ComplexF64, d; v)
export state_psiminus
