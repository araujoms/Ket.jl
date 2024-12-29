"""
    tsirelson_bound(CG::Matrix, scenario::AbstractVecOrTuple, level)

Upper bounds the Tsirelson bound of a multipartite Bell funcional `CG`, written in Collins-Gisin notation.
`scenario` is a tuple detailing the number of inputs and outputs, in the order (oa, ob, ..., ia, ib, ...).
`level` is an integer or string determining the level of the NPA hierarchy.
"""
function tsirelson_bound(CG::Array{T, N}, scenario::AbstractVecOrTuple{<:Integer}, level; solver = Hypatia.Optimizer{_solver_type(T)}) where {T <: Number, N}
    outs = Tuple(scenario[1:N])
    ins = Tuple(scenario[(N + 1):(2 * N)])
    Π = [[QuantumNPA.projector(n, 1:(outs[n] - 1), 1:ins[n]) [QuantumNPA.Id for _ in 1:(outs[n] - 1)]] for n in 1:N]
    cgindex(a, x) = (x .!= (ins .+ 1)) .* (a .+ (x .- 1) .* (outs .- 1)) .+ 1

    bell_functional = QuantumNPA.Polynomial()
    for x in CartesianIndices(ins .+ 1)
        cgiterators = map((a, b, c) -> a == b ? (1:1) : (1:c), x.I, ins .+ 1, outs .- 1)
        for a in Iterators.product(cgiterators...)
            bell_functional += CG[cgindex(a, x.I)...] * prod(Π[n][a[n], x.I[n]] for n in 1:N)
        end
    end

    Q = _npa(_solver_type(T), bell_functional, level; solver)
    return Q
end
export tsirelson_bound

"""
    tsirelson_bound(FC::Matrix, level::Integer)

Upper bounds the Tsirelson bound of a bipartite Bell funcional `FC`, written in full correlation notation.
`level` is an integer or string determining the level of the NPA hierarchy.
"""
function tsirelson_bound(FC::Matrix{T}, level; solver = Hypatia.Optimizer{_solver_type(T)}) where {T <: Number}
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
