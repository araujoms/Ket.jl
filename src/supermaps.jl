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
