using CyclotomicNumbers
using DoubleFloats
using Ket
using Quadmath
using LinearAlgebra
using Test

tests = ["measurements.jl", "norms.jl", "random.jl"]

# SD: I'm no big fan of this for solution. I understand the idea, but it makes it hard to comment out some files when testing a specific set.
# SD: Unless someone has a strong opinion about this, I'll change that in the future.
# MA: Please do.
for test in tests
    include(test)
end
