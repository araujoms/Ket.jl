"""
Toolbox for quantum information, nonlocality, and entanglement.
"""
module Ket

using LinearAlgebra

import Base.AbstractVecOrTuple
import Combinatorics
import Dualization
import GenericLinearAlgebra
import SparseArrays as SA
import Hypatia, Hypatia.Cones
import JuMP
import Nemo
import QuantumNPA

const MOI = JuMP.MOI

include("basic.jl")
include("measurements.jl")
include("states.jl")
include("nonlocal.jl")
include("random.jl")
include("games.jl")
include("incompatibility.jl")
include("entropy.jl")
include("norms.jl")
include("multilinear.jl")
include("channels.jl")
include("entanglement.jl")
include("seesaw.jl")
include("tsirelson.jl")
include("parameterizations.jl")

end # module Ket
