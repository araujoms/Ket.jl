[![Banner](https://dev-ket.github.io/Ket.jl/dev/assets/ket-jl-logo-dark-wide.svg)](https://dev-ket.github.io/Ket.jl/dev/)

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://dev-ket.github.io/Ket.jl/dev/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14674642.svg)](https://doi.org/10.5281/zenodo.14674642)

Ket is a toolbox for quantum information, nonlocality, and entanglement written in the Julia programming language. All its functions are designed to work with generic types, allowing one to use `Int64` or `Float64` for efficiency, or arbitrary precision types when needed. Wherever possible they can be also used for optimization with [JuMP](https://jump.dev/JuMP.jl/stable/). And everything is optimized to the last microsecond.

Highlights:

* Work with multipartite Bell inequalities, computing their local bounds and Tsirelson bounds with `local_bound` and `tsirelson_bound`, and transforming between Collins-Gisin, probability, and correlator representations with `tensor_collinsgisin`, `tensor_probability`, and `tensor_correlation`.
* Work with bipartite entanglement by computing the relative entropy of entanglement, entanglement robustness, or Schmidt number via `entanglement_entropy`, `entanglement_robustness`, and `schmidt_number`. Under the hood these functions use the DPS hierarchy, which is also available in isolation via `_dps_constraints!`.
* Generate MUBs and SIC-POVMs through `mub` and `sic_povm`.
* Generate uniformly-distributed random states, unitaries, and POVMs with `random_state`, `random_unitary`, and `random_povm`.
* Generate well-known families of quantum states, such as the Bell states, the GHZ state, the W state, the Dicke states, and the super-singlet via `state_bell`, `state_ghz`, `state_w`, `state_dicke`, and `state_supersinglet`.
* Work with multilinear algebra via utility functions such as `partial_trace`, `partial_transpose`, and `permute_systems`.
* Generate kets with `ket`.

For the full list of functions see the [documentation](https://dev-ket.github.io/Ket.jl/dev/api/).

## Installation

To use Ket, you must first install Julia by following the instruction in the [official Julia page](https://julialang.org/downloads/).

Ket is a registered Julia package, so it can be installed by typing the following command in the Julia REPL:
```
]add Ket
```
This will install the latest released version. For the development version with the latest updates, use `]add Ket#master` instead.

## Usage

After Ket is installed, you can type
```
using Ket
```
in your REPL.
You will then be able to call any function from the [list of functions](https://dev-ket.github.io/Ket.jl/dev/api/) and to interactively run the examples found in the documentation.

The easiest way to learn how to use a function is by consulting the list of functions or typing `?` followed by the name of the function in the REPL.
You will then be able to see the expected inputs and optional parameters.
For example, by entering `?schmidt_decomposition` you will get:
```
  schmidt_decomposition(ψ::AbstractVector, dims::AbstractVector{<:Integer} = _equal_sizes(ψ))

  Produces the Schmidt decomposition of ψ with subsystem dimensions dims. If the argument dims is omitted
  equally-sized subsystems are assumed. Returns the (sorted) Schmidt coefficients λ and isometries U, V such that
  kron(U', V')*ψ is of Schmidt form.

  Reference: Schmidt decomposition (https://en.wikipedia.org/wiki/Schmidt_decomposition)
```
which is pretty self-explanatory.

There are two other types of arguments which are useful to know. Some functions start with a *type* argument, for example:
```
ket([T=ComplexF64,] i::Integer, d::Integer = 2)
```
This first argument (in square brackets) defines the return type, and it can be used to control the precision of the computations. If you do not know what this is, you can just ignore it, and a sensible default value will be used.

Lastly, some functions have keyword parameters, which are listed after a `;`, for instance:
```
entanglement_robustness(
ρ::AbstractMatrix{T},
dims::AbstractVector{<:Integer} = _equal_sizes(ρ),
n::Integer = 1;
noise::String = "white"
ppt::Bool = true,
inner::Bool = false,
verbose::Bool = false,
solver = Hypatia.Optimizer{_solver_type(T)})
```
These must be set by their names, so for example `entanglement_robustness(ρ; noise = "separable")` is a valid call for any matrix ρ.

If all of this sounds like Greek to you, we recommend you to check out some [basic Julia tutorials](https://julialang.org/learning/) and come back afterwards.

## Related libraries

Julia:

- [QuantumClifford.jl](https://github.com/QuantumSavory/QuantumClifford.jl)
- [QuantumOptics.jl](https://github.com/qojulia/QuantumOptics.jl)
- [QuantumToolbox.jl](https://github.com/qutip/QuantumToolbox.jl)
- [Yao.jl](https://github.com/QuantumBFS/Yao.jl)

Python:
- [toqito](https://github.com/vprusso/toqito)

MATLAB:
- [QETLAB](https://github.com/nathanieljohnston/QETLAB)
