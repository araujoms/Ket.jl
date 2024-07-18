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
```

## Measurements

```@docs
sic_povm
test_sic
test_povm
dilate_povm
povm
mub
test_mub
```

## Nonlocality

```@docs
chsh
cglmp
local_bound
tsirelson_bound
correlation_tensor
probability_tensor
fp2cg
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
