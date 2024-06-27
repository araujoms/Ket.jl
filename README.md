# Ket.jl

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://araujoms.github.io/Ket.jl/dev/)

Toolbox for quantum information, nonlocality, and entanglement.

Highlights are the functions `mub` and `sic_povm`, that produce respectively MUBs and SIC-POVMs with arbitrary precision, `local_bound` that uses a parallelized algorithm to compute the local bound of a Bell inequality, and `partial_trace` and `partial_transpose`, that compute the partial trace and partial transpose in a way that can be used for optimization with [JuMP](https://jump.dev/JuMP.jl/stable/). Also worth mentioning are the functions to produce uniformly-distributed random states, unitaries, and POVMs: `random_state`, `random_unitary`, `random_povm`. And the eponymous `ket`, of course.

For the full list of functions see the [documentation](https://araujoms.github.io/Ket.jl/dev/api/).
