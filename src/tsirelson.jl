"""
    tsirelson_bound(CG::Matrix, scenario::AbstractVecOrTuple, level)

Upper bounds the Tsirelson bound of a multipartite Bell funcional `CG`, written in Collins-Gisin notation.
`scenario` is a tuple detailing the number of inputs and outputs, in the order (oa, ob, ..., ia, ib, ...).
`level` is an integer or string determining the level of the NPA hierarchy.
"""
function tsirelson_bound(
    CG::Array{T,N},
    scenario::AbstractVecOrTuple{<:Integer},
    level;
    verbose = false,
    solver = Hypatia.Optimizer{_solver_type(T)}
) where {T<:Number,N}
    outs = Tuple(scenario[1:N])
    ins = Tuple(scenario[N+1:2N])
    Π = [[[QuantumNPA.Id for _ ∈ 1:outs[n]-1] QuantumNPA.projector(n, 1:outs[n]-1, 1:ins[n])] for n ∈ 1:N]
    cgindex(a, x) = (x .!= 1) .* (a .+ (x .- 2) .* (outs .- 1)) .+ 1

    bell_functional = QuantumNPA.Polynomial()
    for x ∈ CartesianIndices(ins .+ 1)
        for a ∈ CartesianIndices((x.I .!= 1) .* (outs .- 2) .+ 1)
            bell_functional += CG[cgindex(a.I, x.I)...] * prod(Π[n][a.I[n], x.I[n]] for n ∈ 1:N)
        end
    end

    Q = _npa(_solver_type(T), bell_functional, level; verbose, solver)
    return Q
end
export tsirelson_bound
"""
    tsirelson_bound(FC::Matrix, level::Integer; verbose::Bool = false, solver = Hypatia.Optimizer{_solver_type(T)})

Upper bounds the Tsirelson bound of a bipartite Bell funcional `FC`, written in full correlation notation.
`level` is an integer or string determining the level of the NPA hierarchy.
"""
function tsirelson_bound(FC::Array{T,N}, level; verbose = false, solver = Hypatia.Optimizer{_solver_type(T)}) where {T<:Number,N}
    ins = size(FC) .- 1
    O = [[QuantumNPA.Id; QuantumNPA.dichotomic(n, 1:ins[n])] for n ∈ 1:N]

    bell_functional = QuantumNPA.Polynomial()
    for x ∈ CartesianIndices(ins .+ 1)
        bell_functional += FC[x] * prod(O[n][x[n]] for n ∈ 1:N)
    end

    Q = _npa(_solver_type(T), bell_functional, level; verbose, solver)
    return Q
end

function _npa(::Type{T}, functional, level; verbose, solver) where {T<:AbstractFloat}
    model = JuMP.GenericModel{T}()
    moments = QuantumNPA.npa_moment(functional, level)
    dΓ = size(moments)[1]
    JuMP.@variable(model, Z[1:dΓ, 1:dΓ] ∈ JuMP.PSDCone())
    id_matrix = moments[QuantumNPA.Id]
    objective = dot(id_matrix, Z) + functional[QuantumNPA.Id]
    JuMP.@objective(model, Min, objective)
    mons = collect(m for m ∈ QuantumNPA.monomials(functional, moments) if !QuantumNPA.isidentity(m))
    for m ∈ mons
        JuMP.@constraint(model, dot(moments[m], Z) + functional[m] == 0)
    end
    dual_solver = () -> Dualization.DualOptimizer{T}(MOI.instantiate(solver))
    JuMP.set_optimizer(model, dual_solver)
    !verbose && JuMP.set_silent(model)
    JuMP.optimize!(model)
    JuMP.is_solved_and_feasible(model) || throw(error(JuMP.raw_status(model)))
    return JuMP.objective_value(model)
end
