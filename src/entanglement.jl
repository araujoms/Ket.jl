function _equal_sizes(arg::AbstractVecOrMat)
    n = size(arg, 1)
    d = isqrt(n)
    d^2 != n && throw(ArgumentError("Subsystems are not equally-sized, please specify sizes."))
    return [d, d]
end

"""
    schmidt_decomposition(ψ::AbstractVector, dims::AbstractVector{<:Integer} = _equal_sizes(ψ))

Produces the Schmidt decomposition of `ψ` with subsystem dimensions `dims`.
If the argument `dims` is omitted equally-sized subsystems are assumed.
Returns the (sorted) Schmidt coefficients λ and isometries U, V such that kron(U', V')*`ψ` is of Schmidt form.

Reference: [Schmidt decomposition](https://en.wikipedia.org/wiki/Schmidt_decomposition).
"""
function schmidt_decomposition(ψ::AbstractVector, dims::AbstractVector{<:Integer} = _equal_sizes(ψ))
    length(dims) != 2 && throw(ArgumentError("Two subsystem sizes must be specified."))
    m = transpose(reshape(ψ, dims[2], dims[1])) #necessary because the natural reshaping would be row-major, but Julia does it col-major
    U, λ, V = svd(m)
    return λ, U, conj(V)
end
export schmidt_decomposition

"""
    entanglement_entropy(ψ::AbstractVector, dims::AbstractVector{<:Integer} = _equal_sizes(ψ))

Computes the relative entropy of entanglement of a bipartite pure state `ψ` with subsystem dimensions `dims`.
If the argument `dims` is omitted equally-sized subsystems are assumed.
"""
function entanglement_entropy(ψ::AbstractVector, dims::AbstractVector{<:Integer} = _equal_sizes(ψ))
    length(dims) != 2 && throw(ArgumentError("Two subsystem sizes must be specified."))
    max_sys = argmax(dims)
    ρ = partial_trace(ketbra(ψ), max_sys, dims)
    return entropy(ρ)
end
export entanglement_entropy

"""
    entanglement_entropy(ρ::AbstractMatrix, dims::AbstractVector = _equal_sizes(ρ), n::Integer = 1)

Lower bounds the relative entropy of entanglement of a bipartite state `ρ` with subsystem dimensions `dims` using level `n` of the DPS hierarchy.
If the argument `dims` is omitted equally-sized subsystems are assumed.
"""
function entanglement_entropy(ρ::AbstractMatrix{T}, dims::AbstractVector = _equal_sizes(ρ), n::Integer = 1) where {T}
    ishermitian(ρ) || throw(ArgumentError("State needs to be Hermitian"))
    length(dims) != 2 && throw(ArgumentError("Two subsystem sizes must be specified."))

    d = size(ρ, 1)
    is_complex = (T <: Complex)
    Rs = _solver_type(T)
    Ts = is_complex ? Complex{Rs} : Rs
    model = JuMP.GenericModel{Rs}()

    if is_complex
        JuMP.@variable(model, σ[1:d, 1:d], Hermitian)
    else
        JuMP.@variable(model, σ[1:d, 1:d], Symmetric)
    end
    _dps_constraints!(model, σ, dims, n; is_complex)
    JuMP.@constraint(model, tr(σ) == 1)

    vec_dim = Cones.svec_length(Ts, d)
    ρvec = _svec(ρ, Ts)
    σvec = _svec(σ, Ts)

    JuMP.@variable(model, h)
    JuMP.@objective(model, Min, h / log(Rs(2)))
    JuMP.@constraint(model, [h; σvec; ρvec] ∈ Hypatia.EpiTrRelEntropyTriCone{Rs,Ts}(1 + 2 * vec_dim))
    JuMP.set_optimizer(model, Hypatia.Optimizer{Rs})
    JuMP.set_silent(model)
    JuMP.optimize!(model)
    return JuMP.objective_value(model), Hermitian(JuMP.value.(σ))
end

"""
    _svec(M::AbstractMatrix, ::Type{T})

Produces the scaled vectorized version of a Hermitian matrix `M` with coefficient type `T`. The transformation preserves inner products, i.e., ⟨M,N⟩ = ⟨svec(M,T),svec(N,T)⟩.
"""
function _svec(M::AbstractMatrix, ::Type{T}) where {T} #the weird stuff here is to make it work with JuMP variables
    d = size(M, 1)
    R = real(T)
    vec_dim = Cones.svec_length(T, d)
    v = Vector{real(eltype(1 * M))}(undef, vec_dim)
    if T <: Real
        Cones.smat_to_svec!(v, 1 * M, sqrt(R(2)))
    else
        Cones._smat_to_svec_complex!(v, M, sqrt(R(2)))
    end
    return v
end

"""
    _test_entanglement_entropy_qubit(h::Real, ρ::AbstractMatrix, σ::AbstractMatrix)

Checks if `ρ` is indeed a entangled state whose closest separable state is `σ`.

Reference: Miranowicz and Ishizaka, [arXiv:0805.3134](https://arxiv.org/abs/0805.3134)
"""
function _test_entanglement_entropy_qubit(h, ρ, σ)
    R = typeof(h)
    λ, U = eigen(σ)
    g = zeros(R, 4, 4)
    for j ∈ 1:4
        for i ∈ 1:j-1
            g[i, j] = (λ[i] - λ[j]) / log(λ[i] / λ[j])
        end
        g[j, j] = λ[j]
    end
    g = Hermitian(g)
    σT = partial_transpose(σ, 2, [2, 2])
    λ2, U2 = eigen(σT)
    phi = partial_transpose(ketbra(U2[:, 1]), 2, [2, 2])
    G = zero(U)
    for i ∈ 1:4
        for j ∈ 1:4
            G += g[i, j] * ketbra(U[:, i]) * phi * ketbra(U[:, j])
        end
    end
    G = Hermitian(G)
    x = real(pinv(vec(G)) * vec(σ - ρ))
    ρ2 = σ - x * G
    ρ_matches = isapprox(ρ2, ρ; rtol = sqrt(Base.rtoldefault(R)))
    h_matches = isapprox(h, relative_entropy(ρ2, σ); rtol = sqrt(Base.rtoldefault(R)))
    return ρ_matches && h_matches
end

"""
    schmidt_number(
        ρ::AbstractMatrix{T},
        s::Integer = 2,
        dims::AbstractVector{<:Integer} = _equal_sizes(ρ),
        n::Integer = 1;
        ppt::Bool = true,
        verbose::Bool = false,
        solver = Hypatia.Optimizer{_solver_type(T)})

Upper bound on the random robustness of `ρ` such that it has a Schmidt number `s`.

If a state ``ρ`` with local dimensions ``d_A`` and ``d_B`` has Schmidt number ``s``, then there is
a PSD matrix ``ω`` in the extended space ``AA′B′B``, where ``A′`` and ``B^′`` have dimension ``s``,
such that ``ω / s`` is separable  against ``AA′|B′B`` and ``Π† ω Π = ρ``, where ``Π = 1_A ⊗ s ψ^+ ⊗ 1_B``,
and ``ψ^+`` is a non-normalized maximally entangled state. Separabiity is tested with the DPS hierarchy,
with `n` controlling the how many copies of the ``B′B`` subsystem are used.

References:
    Hulpke, Bruss, Lewenstein, Sanpera [arXiv:quant-ph/0401118](https://arxiv.org/abs/quant-ph/0401118)\
Weilenmann, Dive, Trillo, Aguilar, Navascués [arXiv:1912.10056](https://arxiv.org/abs/1912.10056)
"""
function schmidt_number(
    ρ::AbstractMatrix{T},
    s::Integer = 2,
    dims::AbstractVector{<:Integer} = _equal_sizes(ρ),
    n::Integer = 1;
    ppt::Bool = true,
    verbose::Bool = false,
    solver = Hypatia.Optimizer{_solver_type(T)}
) where {T<:Number}
    ishermitian(ρ) || throw(ArgumentError("State must be Hermitian"))
    s ≥ 1 || throw(ArgumentError("Schmidt number must be ≥ 1"))
    if s == 1
        return random_robustness(ρ, dims, n; ppt, verbose, solver)[1]
    end

    is_complex = (T <: Complex)
    wrapper = is_complex ? Hermitian : Symmetric

    V = kron(I(dims[1]), SA.sparse(state_ghz_ket(T, s, 2; coeff = 1)), I(dims[2])) #this is an isometry up to normalization
    lifted_dims = [dims[1] * s, dims[2] * s] # with the ancilla spaces A'B'...

    model = JuMP.GenericModel{_solver_type(T)}()

    JuMP.@variable(model, λ)
    noisy_state = wrapper(ρ + λ * I(size(ρ, 1)))
    JuMP.@objective(model, Min, λ)

    _dps_constraints!(model, noisy_state, lifted_dims, n; ppt, is_complex, isometry = V)
    JuMP.@constraint(model, tr(model[:symmetric_meat]) == s * (tr(ρ) + λ * size(ρ, 1)))

    JuMP.set_optimizer(model, solver)
    !verbose && JuMP.set_silent(model)
    JuMP.optimize!(model)

    if JuMP.is_solved_and_feasible(model)
        return JuMP.objective_value(model)
    else
        return "Something went wrong: $(JuMP.raw_status(model))"
    end
end
export schmidt_number

"""
    random_robustness(
    ρ::AbstractMatrix{T},
    dims::AbstractVector{<:Integer} = _equal_sizes(ρ),
    n::Integer = 1;
    ppt::Bool = true,
    verbose::Bool = false,
    solver = Hypatia.Optimizer{_solver_type(T)})

Lower bounds the random robustness of state `ρ` with subsystem dimensions `dims` using level `n` of the DPS hierarchy. Argument `ppt` indicates whether to include the partial transposition constraints.
"""
function random_robustness(
    ρ::AbstractMatrix{T},
    dims::AbstractVector{<:Integer} = _equal_sizes(ρ),
    n::Integer = 1;
    ppt::Bool = true,
    verbose::Bool = false,
    solver = Hypatia.Optimizer{_solver_type(T)}
) where {T<:Number}
    ishermitian(ρ) || throw(ArgumentError("State must be Hermitian"))

    is_complex = (T <: Complex)
    wrapper = is_complex ? Hermitian : Symmetric

    model = JuMP.GenericModel{_solver_type(T)}()

    JuMP.@variable(model, λ)
    noisy_state = wrapper(ρ + λ * I(size(ρ, 1)))
    _dps_constraints!(model, noisy_state, dims, n; ppt, is_complex)
    JuMP.@objective(model, Min, λ)

    JuMP.set_optimizer(model, solver)
    #JuMP.set_optimizer(model, Dualization.dual_optimizer(solver))    #necessary for acceptable performance with some solvers
    !verbose && JuMP.set_silent(model)
    JuMP.optimize!(model)

    if JuMP.is_solved_and_feasible(model)
        W = JuMP.dual(model[:witness_constraint])
        return JuMP.objective_value(model), W
    else
        return "Something went wrong: $(JuMP.raw_status(model))"
    end
end
export random_robustness

"""
    _dps_constraints!(model::JuMP.GenericModel, ρ::AbstractMatrix, dims::AbstractVector{<:Integer}, n::Integer; ppt::Bool = true, is_complex::Bool = true)

Constrains state `ρ` of dimensions `dims` in JuMP model `model` to respect the DPS constraints of level `n`.

References:
    Doherty, Parrilo, Spedalieri [arXiv:quant-ph/0308032](https://arxiv.org/abs/quant-ph/0308032)
"""
function _dps_constraints!(
    model::JuMP.GenericModel{T},
    ρ::AbstractMatrix,
    dims::AbstractVector{<:Integer},
    n::Integer;
    ppt::Bool = true,
    is_complex::Bool = true,
    isometry::AbstractMatrix = I(size(ρ, 1))
) where {T}
    ishermitian(ρ) || throw(ArgumentError("State must be Hermitian"))

    dA, dB = dims
    ext_dims = [dA; repeat([dB], n)]

    # Dimension of the extension space w/ bosonic symmetries: A dim. + `n` copies of B
    d = dA * binomial(n + dB - 1, n)
    V = kron(I(dA), symmetric_projection(T, dB, n; partial = true)) # Bosonic subspace isometry

    if is_complex
        psd_cone = JuMP.HermitianPSDCone()
        wrapper = Hermitian
    else
        psd_cone = JuMP.PSDCone()
        wrapper = Symmetric
    end

    JuMP.@variable(model, symmetric_meat[1:d, 1:d] ∈ psd_cone)
    lifted = wrapper(V * symmetric_meat * V')
    JuMP.@expression(model, reduced, partial_trace(lifted, 3:n+1, ext_dims))
    JuMP.@constraint(model, witness_constraint, ρ == wrapper(isometry' * reduced * isometry))

    if ppt
        for i ∈ 2:n+1
            JuMP.@constraint(model, partial_transpose(lifted, 2:i, ext_dims) ∈ psd_cone)
        end
    end
end

function _fully_decomposable_witness_constraints!(model, dims, W)
    nparties = length(dims)
    biparts = Combinatorics.partitions(1:nparties, 2)
    dim = prod(dims)

    Ps = [JuMP.@variable(model, [1:dim, 1:dim] ∈ JuMP.HermitianPSDCone()) for _ ∈ 1:length(biparts)]

    JuMP.@constraint(model, tr(W) == 1)
    # this can be used instead of tr(W) = 1 if we want a GME entanglement quantifier (see ref.)
    # [JuMP.@constraint(model, (I(dim) - (W - Ps[i])) ∈ JuMP.HermitianPSDCone()) for i ∈ 1:length(biparts)]

    # constraints for W = Q^{T_M} + P^M:
    for (i, part) ∈ enumerate(biparts)
        JuMP.@constraint(model, Hermitian(partial_transpose(W - Ps[i], part[1], dims)) ∈ JuMP.HermitianPSDCone())
    end
end

function _min_dotprod!(model, ρ, W, solver, verbose)
    JuMP.@variable(model, λ)
    JuMP.@constraint(model, real(dot(ρ, W)) ≤ λ)
    JuMP.@objective(model, Min, λ)

    JuMP.set_optimizer(model, solver)
    !verbose && JuMP.set_silent(model)
    JuMP.optimize!(model)
end

"""
    function ppt_mixture(
    ρ::AbstractMatrix{T},
    dims::AbstractVector{<:Integer};
    verbose::Bool = false,
    solver = Hypatia.Optimizer{_solver_type(T)})

Lower bound on the white noise such that ρ is still a genuinely multipartite entangled state and a GME witness that detects ρ.

The set of GME states is approximated by the set of PPT mixtures, so the entanglement across the bipartitions is decided
with the PPT criterion. If the state is a PPT mixture, returns a 0 matrix instead of a witness.

Reference: Jungnitsch, Moroder, Guehne [arXiv:quant-ph/0401118](https://arxiv.org/abs/quant-ph/0401118)
"""
function ppt_mixture(
    ρ::AbstractMatrix{T},
    dims::AbstractVector{<:Integer};
    verbose::Bool = false,
    solver = Hypatia.Optimizer{_solver_type(T)}
) where {T<:Number}
    dim = size(ρ, 1)
    prod(dims) == size(ρ, 1) || throw(ArgumentError("State dimension does not agree with local dimensions."))

    model = JuMP.GenericModel{_solver_type(T)}()

    JuMP.@variable(model, W[1:dim, 1:dim], Hermitian)

    _fully_decomposable_witness_constraints!(model, dims, W)
    _min_dotprod!(model, ρ, W, solver, verbose)

    if JuMP.is_solved_and_feasible(model)
        JuMP.objective_value(model) ≤ 0 ? W = JuMP.value.(W) : W = zeros(T, size(W))
        return 1 / (1 - dim * JuMP.value.(model[:λ])), Hermitian(W)
    else
        return "Something went wrong: $(JuMP.raw_status(model))"
    end
end

"""
    function ppt_mixture(
    ρ::AbstractMatrix{T},
    dims::AbstractVector{<:Integer},
    obs::AbstractVector{<:AbstractMatrix} = Vector{Matrix}();
    verbose::Bool = false,
    solver = Hypatia.Optimizer{_solver_type(T)})

Lower bound on the white noise such that ρ is still a genuinely multipartite entangled state that
can be detected with a witness using only the operators provided in `obs`, and the values of the coefficients
defining such a witness.

More precisely, if a list of observables ``O_i`` is provided in the parameter `obs`, the witness will be of the form
``∑_i α_i O_i`` and detects ρ only using these observables. For example, using only two-body operators (and lower order)
one can call

```julia-repl
julia> two_body_basis = collect(Iterators.flatten(n_body_basis(i, 3) for i ∈ 0:2))
julia> ppt_mixture(state_ghz(2, 3), [2, 2, 2], two_body_basis)
```

Reference: Jungnitsch, Moroder, Guehne [arXiv:quant-ph/0401118](https://arxiv.org/abs/quant-ph/0401118)
"""
function ppt_mixture(
    ρ::AbstractMatrix{T},
    dims::AbstractVector{<:Integer},
    obs::AbstractVector{<:AbstractMatrix};
    verbose::Bool = false,
    solver = Hypatia.Optimizer{_solver_type(T)}
) where {T<:Number}
    dim = size(ρ, 1)
    prod(dims) == size(ρ, 1) || throw(ArgumentError("State dimension does not agree with local dimensions."))

    model = JuMP.GenericModel{_solver_type(T)}()

    JuMP.@variable(model, w_coeffs[1:length(obs)])
    W = sum(w_coeffs[i] * obs[i] for i ∈ eachindex(w_coeffs))

    _fully_decomposable_witness_constraints!(model, dims, W)
    _min_dotprod!(model, ρ, W, solver, verbose)

    if JuMP.is_solved_and_feasible(model)
        JuMP.objective_value(model) ≤ 0 ? w_coeffs = JuMP.value.(w_coeffs) : w_coeffs = zeros(T, size(w_coeffs))
        return 1 / (1 - dim * JuMP.value.(model[:λ])), w_coeffs
    else
        return "Something went wrong: $(JuMP.raw_status(model))"
    end
end
export ppt_mixture
