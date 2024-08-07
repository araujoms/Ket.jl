module Ket

import Combinatorics
import GenericLinearAlgebra
import LinearAlgebra as LA
import SparseArrays as SA
import Nemo
import JuMP
const MOI = JuMP.MOI
import Hypatia, Hypatia.Cones

include("basic.jl")
include("states.jl")
include("nonlocal.jl")
include("random.jl")
include("games.jl")
include("measurements.jl")
include("entropy.jl")
include("norms.jl")
include("multilinear.jl")
include("supermaps.jl")
include("entanglement.jl")

import Requires
function __init__()
    Requires.@require MATLAB = "10e44e05-a98a-55b3-a45b-ba969058deb6" include("TsirelsonBoundMATLAB.jl")
end

end # module Ket
