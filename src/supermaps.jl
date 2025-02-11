#extract from T the kind of float to be used in the conic solver
_solver_type(::Type{T}) where {T<:AbstractFloat} = T
_solver_type(::Type{Complex{T}}) where {T<:AbstractFloat} = T
_solver_type(::Type{T}) where {T<:Number} = Float64

"""
    applykraus(K::Vector{<:AbstractMatrix}, M::AbstractMatrix)

Applies the CP map given by the Kraus operators `K` to the matrix `M`.
"""
function applykraus(K::Vector{<:AbstractMatrix}, M::AbstractMatrix)
    return sum(Hermitian(Ki * M * Ki') for Ki ∈ K)
end
export applykraus

"""
    channel_bit_flip(rho::AbstractMatrix, p::Real)

The bit flip channel applies Pauli-X with probability `1 − p` (flip from |0> to |1> and vice versa).
"""
function channel_bit_flip(rho::AbstractMatrix, p::Real)
    @assert size(rho) == (2, 2) "This is a qubit channel."
    E0 = [sqrt(p) 0; 0 sqrt(p)]
    E1 = [0 sqrt(1-p); sqrt(1-p) 0]
    return applykraus([E0, E1], rho)
end
export channel_bit_flip

"""
    channel_phase_flip(rho::AbstractMatrix, p::Real)

The phase flip channel applies Pauli-Z with probability `1 − p`.
"""
function channel_phase_flip(rho::AbstractMatrix, p::Real)
    @assert size(rho) == (2, 2) "This is a qubit channel."
    E0 = [sqrt(p) 0; 0 sqrt(p)]
    E1 = [sqrt(1-p) 0; 0 -sqrt(1-p)]
    return applykraus([E0, E1], rho)
end
export channel_phase_flip

"""
    channel_bit_phase_flip(rho::AbstractMatrix, p::Real)

The phase flip channel applies Pauli-Y (=iXY) with probability `1 − p`.
"""
function channel_bit_phase_flip(rho::AbstractMatrix, p::Real)
    @assert size(rho) == (2, 2) "This is a qubit channel."
    E0 = [sqrt(p) 0; 0 sqrt(p)]
    E1 = [0 -im * sqrt(1-p); im * sqrt(1-p) 0]
    return applykraus([E0, E1], rho)
end
export channel_bit_phase_flip

"""
    channel_depolarizing(rho::AbstractMatrix, p::Real)

The depolarizing channel is a single qubit replaced by the completely mixed state with probability 'p'.
"""
function channel_depolarizing(rho::AbstractMatrix, p::Real)
    return white_noise(rho, 1-p)
end
export channel_depolarizing

"""
    channel_amplitude_damping(rho::AbstractMatrix, γ::Real)

The amplitude damping channel describes the effect of dissipation to an environment at zero temperature. 'γ' is the probability of the system to decay to the ground state.
"""
function channel_amplitude_damping(rho::AbstractMatrix, γ::Real)
    @assert size(rho) == (2, 2) "This is a qubit channel."
    E0 = [1 0; 0 sqrt(1-γ)]
    E1 = [0 sqrt(γ); 0 0]
    return applykraus([E0, E1], rho)
end
export channel_amplitude_damping

"""
    channel_generalized_amplitude_damping(rho::AbstractMatrix, p::Real, γ::Real)

The generalized amplitude damping channel describes the effect of dissipation to an environment at finite temperature. 'γ' is the probability of the system to decay to the ground state. '1-p' can be thought of the energy of the stationary state.
"""
function channel_generalized_amplitude_damping(rho::AbstractMatrix, p::Real, γ::Real)
    @assert size(rho) == (2, 2) "This is a qubit channel."
    E0 = [sqrt(p) 0; 0 sqrt(p) * sqrt(1-γ)]
    E1 = [0 sqrt(p) * sqrt(γ); 0 0]
    E2 = [sqrt(1-p) * sqrt(1-γ) 0; 0 sqrt(1-p)]
    E3 = [0 0; sqrt(1-p) * sqrt(γ) 0]
    return applykraus([E0, E1, E2, E3], rho)
end
export channel_generalized_amplitude_damping

"""
    channel_phase_damping(rho::AbstractMatrix, λ::Real)
    
The phase damping channel describes the photon scattering or electron perturbation. 'λ' is the probability being scattered or perturbed (without loss of energy).
"""
function channel_phase_damping(rho::AbstractMatrix, λ::Real) # It can be reformulated as channel_phase_flip(rho, p = (1+√(1 − λ))/2)
    @assert size(rho) == (2, 2) "This is a qubit channel."
    E0 = [1 0; 0 sqrt(1-λ)]
    E1 = [0 0; 0 sqrt(λ)]
    return applykraus([E0, E1], rho)
end
export channel_phase_damping

"""
    choi(K::Vector{<:AbstractMatrix})

Constructs the Choi-Jamiołkowski representation of the CP map given by the Kraus operators `K`.
The convention used is that choi(K) = ∑ᵢⱼ |i⟩⟨j|⊗K|i⟩⟨j|K'
"""
choi(K::Vector{<:AbstractMatrix}) = sum(ketbra(vec(Ki)) for Ki ∈ K)
export choi

"""
    diamond_norm(J::AbstractMatrix, dims::AbstractVector)

Computes the diamond norm of the supermap `J` given in the Choi-Jamiołkowski representation, with subsystem dimensions `dims`.

Reference: [Diamond norm](https://en.wikipedia.org/wiki/Diamond_norm)
"""
function diamond_norm(J::AbstractMatrix{T}, dims::AbstractVector; solver = Hypatia.Optimizer{_solver_type(T)}) where {T}
    ishermitian(J) || throw(ArgumentError("Supermap needs to be Hermitian"))

    is_complex = (T <: Complex)
    psd_cone, wrapper, hermitian_space = _sdp_parameters(is_complex)
    din, dout = dims
    model = JuMP.GenericModel{_solver_type(T)}()
    JuMP.@variable(model, Y[1:din*dout, 1:din*dout] ∈ hermitian_space)
    JuMP.@variable(model, σ[1:din, 1:din] ∈ hermitian_space)
    bigσ = wrapper(kron(σ, I(dout)))
    JuMP.@constraint(model, bigσ - Y ∈ psd_cone)
    JuMP.@constraint(model, bigσ + Y ∈ psd_cone)

    JuMP.@constraint(model, tr(σ) == 1)
    JuMP.@objective(model, Max, real(dot(J, Y)))

    JuMP.set_optimizer(model, solver)
    JuMP.set_silent(model)
    JuMP.optimize!(model)
    return JuMP.objective_value(model)
end
export diamond_norm

"""
    diamond_norm(K::Vector{<:AbstractMatrix})

Computes the diamond norm of the CP map given by the Kraus operators `K`.
"""
function diamond_norm(K::Vector{<:AbstractMatrix})
    dual_to_id = sum(Hermitian(Ki' * Ki) for Ki ∈ K)
    return opnorm(dual_to_id)
end
