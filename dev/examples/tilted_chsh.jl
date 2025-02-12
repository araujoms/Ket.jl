#=
# Tilted CHSH inequality

Given a Bell expression we often want to compute its local bound and the maximum quantum violation.
In Ket this can easily be done using the functions [`local_bound`](@ref), [`seesaw`](@ref) and [`tsirelson_bound`](@ref).

We will use the tilted CHSH inequality as an example
(see [Acín et al.’s “Randomness versus nonlocality and entanglement”](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.108.100402)):

```math
\alpha \langle A_0 \rangle + \langle A_0 B_0 \rangle + \langle A_0 B_1 \rangle + \langle A_1 B_0 \rangle - \langle A_1 B_1 \rangle \overset{C}\leqslant 2+\alpha \overset{Q}\leqslant \sqrt{8+2\alpha^2}
```

In this expression, the ``C`` bound is the local bound (that is, achievable with classical systems),
and the ``Q`` bound is the analytical solution for the maximum violation using quantum systems.
(Of course, in general we do not know the quantum bounds for an arbitrary inequality,
this example is so we can compare the results we will obtain via optimization.)

Let us first define the Bell expression in the full probability representation:
=#

# !!! tip
#     You can convert between behavior representations using the functions [`tensor_probability`](@ref), [`tensor_collinsgisin`](@ref)
#     and [`tensor_correlation`](@ref). They also accept a state and a set of measurements as inputs, returning the corresponding behavior.

function tilted_chsh(α)
    ## in correlator notation, the tilted CHSH is:
    corr = [0  α  0;
            0  1  1;
            0  1 -1]
    return tensor_probability(corr)
end
println() #hide

#=
Computing the lower bound amounts to finding the maximum of the expression over all deterministic strategies.
This can be done using the [`local_bound`](@ref) function, which can be called on any expression written in correlation
or full probability format.
=#

using Ket

## we take 10 different values of α
α = LinRange(0, 1, 10)
## the `.` operator applies the function to each element of the vector
local_bounds = local_bound.(tilted_chsh.(α))

#=
For the quantum value, we can:

1. Obtain lower bounds (with a quantum realization in a given dimension) using the [`seesaw`](@ref) function,
    whose inputs are an inequality in the Collins-Gisin representation, a vector specifying the scenario, and the dimension.
    Since the seesaw algorithm can get trapped in local maxima, it is recommended to run it multiple times and select the best shot.
    This is automated via the optional last argument `n_shots`.

2. Obtain upper bounds using the [`tsirelson_bound`](@ref) function, which is based on the NPA hierarchy.
    It takes an inequality in the Collins-Gisin or full probability representation, a vector specifying the scenario
    (the number of outcomes and inputs per party), and the level of the NPA hierarchy.
=#

tilted_chsh_cg(α) = tensor_collinsgisin(tilted_chsh(α))

## the first output of seesaw is the bound
quantum_lbounds = [seesaw(tilted_chsh_cg(αi), [2, 2, 2, 2], 2, 100)[1] for αi in α]

quantum_ubounds_l1 = [tsirelson_bound(tilted_chsh_cg(αi), [2, 2, 2, 2], 1) for αi in α]
quantum_ubounds_l2 = [tsirelson_bound(tilted_chsh_cg(αi), [2, 2, 2, 2], 2) for αi in α]
println() #hide

#=
To visualize the bounds, we can plot the results and compare them to the analytical solutions:
=#

using Plots

xs = LinRange(0, 1, 100)

plt = plot(xs, (x -> 2 + x).(xs), label = "Local bound (analytical)", linewidth = 2.5)
scatter!(α, local_bounds, label = "Local bound (numerical)", markersize = 5, markershape = :circle)

plot!(xs, (x -> sqrt(8 + 2x^2)).(xs), label = "Quantum bound (analytical)", linewidth = 2.5)
scatter!(α, quantum_lbounds, label = "Quantum bound (seesaw)", markersize = 4, markershape = :square)
scatter!(α, quantum_ubounds_l1, label = "Quantum bound (NPA 1)", markersize = 5, markershape = :diamond)
scatter!(α, quantum_ubounds_l2, label = "Quantum bound (NPA 2)", markersize = 4, markershape = :utriangle)
plot!(xlabel = "α", title = "α⟨A₀⟩ + CHSH")
