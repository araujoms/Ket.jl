"""
    incompatibility_robustness_depolarizing(A::Vector{Measurement{<:Number}})

Computes the incompatibility depolarizing robustness of the measurements in the vector `A`.

Reference: Designolle, Farkas, Kaniewski, [arXiv:1906.00448](https://arxiv.org/abs/1906.00448)
"""
function incompatibility_robustness_depolarizing(
    A::Vector{Measurement{T}};
    verbose = false,
    return_parent = false,
    solver = Hypatia.Optimizer{_solver_type(T)}
) where {T<:Number}
    d, o, m = _measurements_parameters(A)
    stT = _solver_type(T)
    model = JuMP.GenericModel{stT}()
    if T <: Complex
        X = [[JuMP.@variable(model, [1:d, 1:d], Hermitian) for a in 1:o[x]] for x in 1:m]
        con = JuMP.@constraint(model, [j in CartesianIndices(o)], sum(X[x][j.I[x]] for x in 1:m) in JuMP.HermitianPSDCone())
    else
        X = [[JuMP.@variable(model, [1:d, 1:d], Symmetric) for a in 1:o[x]] for x in 1:m]
        con = JuMP.@constraint(model, [j in CartesianIndices(o)], sum(X[x][j.I[x]] for x in 1:m) in JuMP.PSDCone())
    end
    obj = zero(JuMP.GenericAffExpr{stT,JuMP.GenericVariableRef{stT}})
    low = zero(JuMP.GenericAffExpr{stT,JuMP.GenericVariableRef{stT}})
    JuMP.add_to_expression!(obj, 1)
    for x in 1:m
        for a in 1:o[x]
            JuMP.add_to_expression!(obj, LA.tr(X[x][a] * A[x][a]))
            JuMP.add_to_expression!(low, (LA.tr(A[x][a]) / d) * LA.tr(X[x][a]))
        end
    end
    JuMP.@objective(model, Min, obj)
    JuMP.@constraint(model, obj ≥ low)

    JuMP.set_optimizer(model, solver)
    !verbose && JuMP.set_silent(model)
    JuMP.optimize!(model)
    if JuMP.is_solved_and_feasible(model)
        η = JuMP.objective_value(model)
        if return_parent && JuMP.has_duals(model)
            # the parent povm is best represented in the tensor format as it has many outcomes
            G = zeros(T, d, d, o...)
            for (j, c) in zip(CartesianIndices(o), con)
                G[:, :, j] .= JuMP.dual(c)
            end
            return η, G
        else
            return η #, [[JuMP.value.(X[x][a]) for a in 1:o[x]] for x in 1:m]
        end
    else
        return "Something went wrong: $(JuMP.raw_status(model))"
    end
end
export incompatibility_robustness_depolarizing
