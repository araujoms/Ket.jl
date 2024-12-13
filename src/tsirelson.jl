"""
    tsirelson_bound(CG::Matrix, scenario::AbstractVecOrTuple, level::Integer)

Upper bounds the Tsirelson bound of a bipartite Bell funcional game `CG`, written in Collins-Gisin notation.
`scenario` is a vector detailing the number of inputs and outputs, in the order [oa, ob, ia, ib].
`level` is an integer determining the level of the NPA hierarchy.
"""
function tsirelson_bound(CG::Matrix{T}, scenario::AbstractVecOrTuple{<:Integer}, level::Integer; solver = Hypatia.Optimizer{_solver_type(T)}) where {T <: Number}
    oa, ob, ia, ib = scenario
    A = QuantumNPA.projector(1, 1:(oa - 1), 1:ia)
    B = QuantumNPA.projector(2, 1:(ob - 1), 1:ib)

    aind(a, x) = 1 + a + (x - 1) * (oa - 1)
    bind(b, y) = 1 + b + (y - 1) * (ob - 1)

    bell_functional = sum(CG[aind(a, x), bind(b, y)] * A[a, x] * B[b, y] for a in 1:(oa - 1), b in 1:(ob - 1), x in 1:ia, y in 1:ib)
    bell_functional += sum(CG[aind(a, x), 1] * A[a, x] for a in 1:(oa - 1), x in 1:ia)
    bell_functional += sum(CG[1, bind(b, y)] * B[b, y] for b in 1:(ob - 1), y in 1:ib)
    bell_functional += CG[1, 1] * QuantumNPA.Id

    Q = _npa(_solver_type(T), bell_functional, level; solver)
    return Q
end
export tsirelson_bound

"""
    tsirelson_bound_fc(FC::Matrix, level::Integer)

Upper bounds the Tsirelson bound of a bipartite Bell funcional game `FC`, written in full correlation notation.
`level` is an integer determining the level of the NPA hierarchy.
"""
function tsirelson_bound_fc(FC::Matrix{T}, level::Integer; solver = Hypatia.Optimizer{_solver_type(T)}) where {T <: Number}
    ia, ib = size(FC) .- 1
    A = QuantumNPA.dichotomic(1, 1:ia)
    B = QuantumNPA.dichotomic(2, 1:ib)

    bell_functional = sum(FC[x + 1, y + 1] * A[x] * B[y] for x in 1:ia, y in 1:ib)
    bell_functional += sum(FC[x + 1, 1] * A[x] for x in 1:ia)
    bell_functional += sum(FC[1, y + 1] * B[y] for y in 1:ib)
    bell_functional += FC[1, 1] * QuantumNPA.Id

    Q = _npa(_solver_type(T), bell_functional, level; solver)
    return Q
end
export tsirelson_bound_fc

function _npa(::Type{T}, functional, level; solver) where {T <: AbstractFloat}
    model = JuMP.GenericModel{T}()
    moments = QuantumNPA.npa_moment(functional, level)
    dΓ = size(moments)[1]
    JuMP.@variable(model, Z[1:dΓ, 1:dΓ] in JuMP.PSDCone())
    id_matrix = moments[QuantumNPA.Id]
    objective = dot(id_matrix, Z) + functional[QuantumNPA.Id]
    JuMP.@objective(model, Min, objective)
    mons = collect(m for m in QuantumNPA.monomials(functional, moments) if !QuantumNPA.isidentity(m))
    for m in mons
        JuMP.@constraint(model, dot(moments[m], Z) + functional[m] == 0)
    end
    dual_solver = () -> Dualization.DualOptimizer{T}(MOI.instantiate(solver))
    JuMP.set_optimizer(model, dual_solver)
    JuMP.set_silent(model)
    JuMP.optimize!(model)
    return JuMP.objective_value(model)
end
