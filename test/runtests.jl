using Ket
using Test
using DoubleFloats
using Quadmath

tests = ["norms.jl"]

for test in tests
    include(test);
end
