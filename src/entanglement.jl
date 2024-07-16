function _equal_sizes(f, arg)
    n = size(arg, 1)
    d = isqrt(n)
    d^2 != n && throw(ArgumentError("Subsystems are not equally-sized, please specify sizes."))
    return f(arg, [d, d])
end

"""
    schmidt_decomposition(ψ::AbstractVector, dims::AbstractVector{<:Integer})
    
Produces the Schmidt decomposition of `ψ` with subsystem dimensions `dims`. Returns the (sorted) Schmidt coefficients λ and isometries U, V such that
kron(U', V')*`ψ` is of Schmidt form.

Reference: [Schmidt decomposition](https://en.wikipedia.org/wiki/Schmidt_decomposition).
"""
function schmidt_decomposition(ψ::AbstractVector, dims::AbstractVector{<:Integer})
    length(dims) != 2 && throw(ArgumentError("Two subsystem sizes must be specified."))
    m = transpose(reshape(ψ, dims[2], dims[1])) #necessary because the natural reshaping would be row-major, but Julia does it col-major
    U, λ, V = LA.svd(m)
    return λ, U, conj(V)
end
export schmidt_decomposition

"""
    schmidt_decomposition(ψ::AbstractVector)
    
Produces the Schmidt decomposition of `ψ` assuming equally-sized subsystems. Returns the (sorted) Schmidt coefficients λ and isometries U, V such that kron(U', V')*`ψ` is of Schmidt form.

Reference: [Schmidt decomposition](https://en.wikipedia.org/wiki/Schmidt_decomposition).
"""
schmidt_decomposition(ψ::AbstractVector) = _equal_sizes(schmidt_decomposition, ψ)

"""
    entanglement_entropy(ψ::AbstractVector, dims::AbstractVector{<:Integer})

Computes the relative entropy of entanglement of a bipartite pure state `ψ` with subsystem dimensions `dims`.
"""
function entanglement_entropy(ψ::AbstractVector, dims::AbstractVector{<:Integer})
    length(dims) != 2 && throw(ArgumentError("Two subsystem sizes must be specified."))
    max_sys = argmax(dims)
    ρ = partial_trace(ketbra(ψ), max_sys, dims)
    return entropy(ρ)
end
export entanglement_entropy

"""
    entanglement_entropy(ψ::AbstractVector)

Computes the relative entropy of entanglement of a bipartite pure state `ψ` assuming equally-sized subsystems.
"""
entanglement_entropy(ψ::AbstractVector) = _equal_sizes(entanglement_entropy, ψ)

"""
    entanglement_entropy(ρ::AbstractMatrix, dims::AbstractVector)

Lower bounds the relative entropy of entanglement of a bipartite state `ρ` with subsystem dimensions `dims`.
"""
function entanglement_entropy(ρ::AbstractMatrix{T}, dims::AbstractVector) where {T}
    LA.ishermitian(ρ) || throw(ArgumentError("State needs to be Hermitian"))
    length(dims) != 2 && throw(ArgumentError("Two subsystem sizes must be specified."))

    d = size(ρ, 1)
    is_complex = (T <: Complex)
    Rs = _solver_type(T)
    Ts = is_complex ? Complex{Rs} : Rs
    model = JuMP.GenericModel{Rs}()

    if is_complex
        JuMP.@variable(model, σ[1:d, 1:d], Hermitian)
        σT = partial_transpose(σ, 2, dims)
        JuMP.@constraint(model, σT in JuMP.HermitianPSDCone())
    else
        JuMP.@variable(model, σ[1:d, 1:d], Symmetric)
        σT = partial_transpose(σ, 2, dims)
        JuMP.@constraint(model, σT in JuMP.PSDCone())
    end
    JuMP.@constraint(model, LA.tr(σ) == 1)

    vec_dim = Cones.svec_length(Ts, d)
    ρvec = _svec(ρ, Ts)
    σvec = _svec(σ, Ts)

    JuMP.@variable(model, h)
    JuMP.@objective(model, Min, h / log(Rs(2)))
    JuMP.@constraint(model, [h; σvec; ρvec] in Hypatia.EpiTrRelEntropyTriCone{Rs,Ts}(1 + 2 * vec_dim))
    JuMP.set_optimizer(model, Hypatia.Optimizer{Rs})
    JuMP.set_attribute(model, "verbose", false)
    JuMP.optimize!(model)
    return JuMP.objective_value(model), JuMP.value.(σ)
end

"""
    entanglement_entropy(ρ::AbstractMatrix)

Lower bounds the relative entropy of entanglement of a bipartite state `ρ` assuming equally-sized subsystems.
"""
entanglement_entropy(ρ::AbstractMatrix) = _equal_sizes(entanglement_entropy, ρ)

"""
    _svec(M::AbstractMatrix, ::Type{R})

Produces the scaled vectorized version of a Hermitian matrix `M` with coefficient type `R`. The transformation preserves inner products, i.e., ⟨M,N⟩ = ⟨svec(M,R),svec(N,R)⟩.
"""
function _svec(M::AbstractMatrix, ::Type{R}) where {R} #the weird stuff here is to make it work with JuMP variables
    d = size(M, 1)
    T = real(R)
    vec_dim = Cones.svec_length(R, d)
    v = Vector{real(eltype(1 * M))}(undef, vec_dim)
    if R <: Real
        Cones.smat_to_svec!(v, 1 * M, sqrt(T(2)))
    else
        Cones._smat_to_svec_complex!(v, M, sqrt(T(2)))
    end
    return v
end
