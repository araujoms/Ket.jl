[![Banner](https://araujoms.github.io/Ket.jl/dev/assets/ket-jl-logo-dark-wide.svg)](https://araujoms.github.io/Ket.jl/dev/)

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://araujoms.github.io/Ket.jl/dev/)

Ket is a toolbox for quantum information, nonlocality, and entanglement written in the Julia programming language. All its functions are designed to work with generic types, allowing one to use `Int64` or `Float64` for efficiency, or arbitrary precision types when needed. Wherever possible they can be also used for optimization with [JuMP](https://jump.dev/JuMP.jl/stable/). And everything is optimized to the last microsecond.

Highlights:

* Work with multipartite Bell inequalities, computing their local bounds and Tsirelson bounds with `local_bound` and `tsirelson_bound`, and transforming between Collins-Gisin, probability, and correlator representations with `tensor_collinsgisin`, `tensor_probability`, and `tensor_correlation`.
* Work with bipartite entanglement by computing the relative entropy of entanglement, random robustness, or Schmidt number via `entanglement_entropy`, `random_robustness`, and `schmidt_number`. Under the hood these functions use the DPS hierarchy, which is also available in isolation via `_dps_constraints!`.
* Generate MUBs and SIC-POVMs through `mub` and `sic_povm`.
* Generate uniformly-distributed random states, unitaries, and POVMs with `random_state`, `random_unitary`, and `random_povm`.
* Generate well-known families of quantum states, such as the Bell states, the GHZ state, the W state, and the super-singlet via `state_bell`, `state_ghz`, `state_w`, and `state_supersinglet`.
* Work with multilinear algebra via utility functions such as `partial_trace`, `partial_transpose`, and `permute_systems`.
* Generate kets with `ket`.

For the full list of functions see the [documentation](https://araujoms.github.io/Ket.jl/dev/api/).
