using Ket
using Test


tests = ["norms.jl"]

for test in tests
    include(test);
end