# List of functions

## Basic

```@docs
ket
ketbra
proj
shift
clock
pauli
gell_mann
gell_mann!
partial_trace
partial_transpose
permute_systems!
permute_systems
cleanup!
symmetric_projection
orthonormal_range
permutation_matrix
n_body_basis
```

## Entropy

```@docs
entropy
binary_entropy
relative_entropy
binary_relative_entropy
conditional_entropy
```

## Entanglement

```@docs
schmidt_decomposition
entanglement_entropy
random_robustness
schmidt_number
ppt_mixture
```

## Measurements

```@docs
sic_povm
test_sic
test_povm
dilate_povm
povm
tensor_to_povm
povm_to_tensor
mub
test_mub
```

## Incompatibility

```@docs
incompatibility_robustness
incompatibility_robustness_depolarizing
incompatibility_robustness_random
incompatibility_robustness_probabilistic
incompatibility_robustness_jointly_measurable
incompatibility_robustness_generalized
```

## Nonlocality

```@docs
chsh
cglmp
inn22
local_bound
tsirelson_bound
seesaw
correlation_tensor
probability_tensor
fp2cg
cg2fp
```

## Norms

```@docs
trace_norm
kyfan_norm
schatten_norm
diamond_norm
```

## Random

```@docs
random_state
random_state_ket
random_unitary
random_povm
random_probability
```

## States

```@docs
state_phiplus_ket
state_phiplus
isotropic
state_psiminus_ket
state_psiminus
state_super_singlet_ket
state_super_singlet
state_ghz_ket
state_ghz
state_w_ket
state_w
white_noise
white_noise!
```

## Supermaps

```@docs
choi
```

## Internal functions

```@docs
Ket._partition
Ket._fiducial_WH
Ket._idx
Ket._tidx
Ket._idxperm
```
