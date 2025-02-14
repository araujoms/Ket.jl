"""
    incompatibility_robustness(A::Vector{Measurement{<:Number}}; noise::String = "general")

Computes the incompatibility robustness of the measurements in the vector `A`.
Depending on the noise model chosen, the second argument can be
"depolarizing",
"random",
"probabilistic",
"jointly_measurable",
or "general" (default),
see the corresponding functions below.
Returns the parent POVM if `return_parent = true`.

Reference: Designolle, Farkas, Kaniewski, [arXiv:1906.00448](https://arxiv.org/abs/1906.00448)
"""
function incompatibility_robustness(
    A::Vector{Measurement{T}};
    noise::String = "general",
    verbose = false,
    return_parent = false,
    solver = Hypatia.Optimizer{_solver_type(T)}
) where {T<:Number}
    @assert noise ∈ ["depolarizing", "random", "probabilistic", "jointly_measurable", "general"]
    d, o, m = _measurements_parameters(A)
    is_complex = T <: Complex
    psd_cone, wrapper, hermitian_space = _sdp_parameters(is_complex)
    stT = _solver_type(T)
    model = JuMP.GenericModel{stT}()

    # variables
    X = [[JuMP.@variable(model, [1:d, 1:d] ∈ hermitian_space) for a ∈ 1:o[x]] for x ∈ 1:m]
    if noise ∈ ["jointly_measurable", "general"]
        N = JuMP.@variable(model, [1:d, 1:d] ∈ hermitian_space)
    end
    if noise == "probabilistic"
        ξ = JuMP.@variable(model, [1:m])
    end

    # constraints
    jumpT = typeof(real(1 * X[1][1][1]))
    lhs = zero(jumpT)
    rhs = zero(jumpT)
    if noise ∈ ["depolarizing", "random", "probabilistic"]
        con = JuMP.@constraint(model, [j ∈ CartesianIndices(o)], sum(X[x][j.I[x]] for x ∈ 1:m) ∈ psd_cone)
        JuMP.add_to_expression!(lhs, 1)
    else
        con = JuMP.@constraint(model, [j ∈ CartesianIndices(o)], N - sum(X[x][j.I[x]] for x ∈ 1:m) ∈ psd_cone)
        if noise == "jointly_measurable"
            JuMP.@constraint(model, [j ∈ CartesianIndices(o)], sum(X[x][j.I[x]] for x ∈ 1:m) ∈ psd_cone)
        end
        JuMP.add_to_expression!(rhs, 1)
    end
    for x ∈ 1:m
        for a ∈ 1:o[x]
            JuMP.add_to_expression!(lhs, dot(X[x][a], A[x][a]))
            if noise == "depolarizing"
                JuMP.add_to_expression!(rhs, (tr(A[x][a]) / d) * tr(X[x][a]))
            elseif noise == "random"
                JuMP.add_to_expression!(rhs, (1 / o[x]) * tr(X[x][a]))
            elseif noise == "probabilistic"
                JuMP.@constraint(model, ξ[x] ≥ real(tr(X[x][a])))
            elseif noise == "general"
                JuMP.@constraint(model, X[x][a] ∈ psd_cone)
            end
        end
        if noise == "probabilistic"
            JuMP.add_to_expression!(rhs, ξ[x])
        end
    end
    JuMP.@constraint(model, lhs ≥ rhs)

    # objetive function
    if noise ∈ ["depolarizing", "random", "probabilistic"]
        JuMP.@objective(model, Min, lhs)
    else
        JuMP.@objective(model, Min, real(tr(N)))
    end

    # call of the solver
    JuMP.set_optimizer(model, solver)
    !verbose && JuMP.set_silent(model)
    JuMP.optimize!(model)
    JuMP.is_solved_and_feasible(model) || throw(error(JuMP.raw_status(model)))
    η = JuMP.objective_value(model)
    if return_parent && JuMP.has_duals(model)
        # the parent POVM is best represented in the tensor format as it has many outcomes
        G = zeros(T, d, d, o...)
        for (j, c) ∈ zip(CartesianIndices(o), con)
            G[:, :, j] .= JuMP.dual(c)
        end
        cleanup!(G)
        return η, G
    else
        return η #, [[JuMP.value.(X[x][a]) for a ∈ 1:o[x]] for x ∈ 1:m]
    end
end
export incompatibility_robustness

"""
    incompatibility_robustness_depolarizing(A::Vector{Measurement{<:Number}})

Computes the incompatibility depolarizing robustness of the measurements in the vector `A`.

Reference: Designolle, Farkas, Kaniewski, [arXiv:1906.00448](https://arxiv.org/abs/1906.00448)
"""
function incompatibility_robustness_depolarizing(A::Vector{Measurement{T}}; kwargs...) where {T<:Number}
    return incompatibility_robustness(A; noise = "depolarizing", kwargs...)
end
export incompatibility_robustness_depolarizing

"""
    incompatibility_robustness_random(A::Vector{Measurement{<:Number}})

Computes the incompatibility random robustness of the measurements in the vector `A`.

Reference: Designolle, Farkas, Kaniewski, [arXiv:1906.00448](https://arxiv.org/abs/1906.00448)
"""
function incompatibility_robustness_random(A::Vector{Measurement{T}}; kwargs...) where {T<:Number}
    return incompatibility_robustness(A; noise = "random", kwargs...)
end
export incompatibility_robustness_random

"""
    incompatibility_robustness_probabilistic(A::Vector{Measurement{<:Number}})

Computes the incompatibility probabilistic robustness of the measurements in the vector `A`.

Reference: Designolle, Farkas, Kaniewski, [arXiv:1906.00448](https://arxiv.org/abs/1906.00448)
"""
function incompatibility_robustness_probabilistic(A::Vector{Measurement{T}}; kwargs...) where {T<:Number}
    return incompatibility_robustness(A; noise = "probabilistic", kwargs...)
end
export incompatibility_robustness_probabilistic

"""
    incompatibility_robustness_jointly_measurable(A::Vector{Measurement{<:Number}})

Computes the incompatibility jointly measurable robustness of the measurements in the vector `A`.

Reference: Designolle, Farkas, Kaniewski, [arXiv:1906.00448](https://arxiv.org/abs/1906.00448)
"""
function incompatibility_robustness_jointly_measurable(A::Vector{Measurement{T}}; kwargs...) where {T<:Number}
    return incompatibility_robustness(A; noise = "jointly_measurable", kwargs...)
end
export incompatibility_robustness_jointly_measurable

"""
    incompatibility_robustness_generalized(A::Vector{Measurement{<:Number}})

Computes the incompatibility generalized robustness of the measurements in the vector `A`.

Reference: Designolle, Farkas, Kaniewski, [arXiv:1906.00448](https://arxiv.org/abs/1906.00448)
"""
function incompatibility_robustness_generalized(A::Vector{Measurement{T}}; kwargs...) where {T<:Number}
    return incompatibility_robustness(A; noise = "general", kwargs...)
end
export incompatibility_robustness_generalized
