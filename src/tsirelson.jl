"""
    tsirelson_bound(CG::Array, scenario::Tuple, level; verbose::Bool = false, dualize::Bool = false, solver = Hypatia.Optimizer{_solver_type(T)})

Upper bounds the Tsirelson bound of a multipartite Bell funcional `CG`, written in Collins-Gisin notation.
`scenario` is a tuple detailing the number of inputs and outputs, in the order (oa, ob, ..., ia, ib, ...).
`level` is an integer or a string like "1 + A B" determining the level of the NPA hierarchy.
`verbose` determines whether solver output is printed.
`dualize` determines whether the dual problem is solved instead. WARNING: This is critical for performance, and the correct choice depends on the solver.
"""
function tsirelson_bound(
    CG::Array{T,N},
    scenario::Tuple,
    level;
    verbose::Bool = false,
    dualize::Bool = false,
    solver = Hypatia.Optimizer{_solver_type(T)}
) where {T<:Number,N}
    outs = scenario[1:N]
    ins = scenario[N+1:2N]
    Π = [[[QuantumNPA.Id for _ ∈ 1:outs[n]-1] QuantumNPA.projector(n, 1:outs[n]-1, 1:ins[n])] for n ∈ 1:N]
    cgindex(a, x) = (x .!= 1) .* (a .+ (x .- 2) .* (outs .- 1)) .+ 1
    behaviour_operator = Array{QuantumNPA.Monomial,N}(undef, size(CG))
    for x ∈ CartesianIndices(ins .+ 1)
        for a ∈ CartesianIndices((x.I .!= 1) .* (outs .- 2) .+ 1)
            behaviour_operator[cgindex(a.I, x.I)...] = prod(Π[n][a.I[n], x.I[n]] for n ∈ 1:N)
        end
    end
    Q, behaviour = _npa(_solver_type(T).(CG), behaviour_operator, level; verbose, solver, dualize)
    return Q, behaviour
end
export tsirelson_bound

"""
    tsirelson_bound(FC::Array, level; verbose::Bool = false, dualize::Bool = false, solver = Hypatia.Optimizer{_solver_type(T)})

Upper bounds the Tsirelson bound of a multipartite Bell funcional `FC`, written in correlator notation.
`level` is an integer or a string like "1 + A B" determining the level of the NPA hierarchy.
`verbose` determines whether solver output is printed.
`dualize` determines whether the dual problem is solved instead. WARNING: This is critical for performance, and the correct choice depends on the solver.
"""
function tsirelson_bound(
    FC::Array{T,N},
    level;
    verbose::Bool = false,
    dualize::Bool = false,
    solver = Hypatia.Optimizer{_solver_type(T)}
) where {T<:Number,N}
    ins = size(FC) .- 1
    O = [[QuantumNPA.Id; QuantumNPA.dichotomic(n, 1:ins[n])] for n ∈ 1:N]

    behaviour_operator = Array{QuantumNPA.Monomial,N}(undef, size(FC))
    for x ∈ CartesianIndices(ins .+ 1)
        behaviour_operator[x] = prod(O[n][x[n]] for n ∈ 1:N)
    end

    Q, behaviour = _npa(_solver_type(T).(FC), behaviour_operator, level; verbose, solver, dualize)
    return Q, behaviour
end

function _npa(functional::Array{T,N}, behaviour_operator, level; verbose, solver, dualize) where {T<:AbstractFloat,N}
    model = JuMP.GenericModel{T}()
    Γ_basis = QuantumNPA.npa_moment(behaviour_operator, level)
    monomials = setdiff(QuantumNPA.monomials(Γ_basis), [QuantumNPA.Id])
    JuMP.@variable(model, var[monomials])
    dΓ = size(Γ_basis)[1]
    Γ = Matrix{typeof(1 * first(var))}(undef, dΓ, dΓ)
    for i ∈ eachindex(Γ)
        Γ[i] = 0
    end
    Γ .+= Γ_basis[QuantumNPA.Id]
    for m ∈ monomials
        _jump_muladd!(Γ, Γ_basis[m], var[m])
    end
    JuMP.@constraint(model, Γ ∈ JuMP.PSDCone())
    behaviour = Array{typeof(1 * first(var)),N}(undef, size(behaviour_operator))
    behaviour[1] = 1
    for i ∈ 2:length(behaviour)
        behaviour[i] = var[behaviour_operator[i]]
    end
    objective = dot(functional, behaviour)
    JuMP.@objective(model, Max, objective)
    if dualize
        JuMP.set_optimizer(model, Dualization.dual_optimizer(solver; coefficient_type = T))
    else
        JuMP.set_optimizer(model, solver)
    end
    !verbose && JuMP.set_silent(model)
    JuMP.optimize!(model)
    JuMP.is_solved_and_feasible(model) || throw(error(JuMP.raw_status(model)))
    return JuMP.objective_value(model), JuMP.value.(behaviour)
end

function _jump_muladd!(G, A::SA.SparseMatrixCSC, jumpvar)
    for j ∈ 1:size(A, 2)
        for k ∈ SA.nzrange(A, j)
            JuMP.add_to_expression!(G[SA.rowvals(A)[k], j], SA.nonzeros(A)[k], jumpvar)
        end
    end
    return G
end
