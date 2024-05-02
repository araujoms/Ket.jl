module Ket

import Combinatorics
import CyclotomicNumbers as CN
import GenericLinearAlgebra
import LinearAlgebra as LA
import Nemo

include("basic.jl")
include("states.jl")
include("nonlocal.jl")
include("random.jl")
include("games.jl")
include("measurements.jl")
include("entropy.jl")
include("norms.jl")

import Requires
function __init__()
    Requires.@require MATLAB = "10e44e05-a98a-55b3-a45b-ba969058deb6" include("TsirelsonBoundMATLAB.jl")
end

end # module Ket
