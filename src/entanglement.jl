function _equal_sizes(arg::AbstractVecOrMat)
    n = size(arg, 1)
    d = isqrt(n)
    d^2 != n && throw(ArgumentError("Subsystems are not equally-sized, please specify sizes."))
    return [d, d]
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
schmidt_decomposition(ψ::AbstractVector) = schmidt_decomposition(ψ, _equal_sizes(ψ))

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
entanglement_entropy(ψ::AbstractVector) = entanglement_entropy(ψ, _equal_sizes(ψ))

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
    return JuMP.objective_value(model), LA.Hermitian(JuMP.value.(σ))
end

"""
    entanglement_entropy(ρ::AbstractMatrix)

Lower bounds the relative entropy of entanglement of a bipartite state `ρ` assuming equally-sized subsystems.
"""
entanglement_entropy(ρ::AbstractMatrix) = entanglement_entropy(ρ, _equal_sizes(ρ))

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

"""
    _test_entanglement_entropy_qubit(h::Real, ρ::AbstractMatrix, σ::AbstractMatrix)

Tests whether `ρ` is indeed a entangled state whose closest separable state is `σ`.

Reference: Miranowicz and Ishizaka, [arXiv:0805.3134](https://arxiv.org/abs/0805.3134)
"""
function _test_entanglement_entropy_qubit(h, ρ, σ)
    R = typeof(h)
    λ, U = LA.eigen(σ)
    g = zeros(R, 4, 4)
    for j = 1:4
        for i = 1:j-1
            g[i, j] = (λ[i] - λ[j]) / log(λ[i] / λ[j])
        end
        g[j, j] = λ[j]
    end
    g = LA.Hermitian(g)
    σT = partial_transpose(σ, 2, [2, 2])
    λ2, U2 = LA.eigen(σT)
    phi = partial_transpose(ketbra(U2[:, 1]), 2, [2, 2])
    G = zero(U)
    for i = 1:4
        for j = 1:4
            G += g[i, j] * ketbra(U[:, i]) * phi * ketbra(U[:, j])
        end
    end
    G = LA.Hermitian(G)
    x = real(LA.pinv(vec(G)) * vec(σ - ρ))
    ρ2 = σ - x * G
    ρ_matches = isapprox(ρ2, ρ; rtol = sqrt(Base.rtoldefault(R)))
    h_matches = isapprox(h, relative_entropy(ρ2, σ); rtol = sqrt(Base.rtoldefault(R)))
    return ρ_matches && h_matches
end
