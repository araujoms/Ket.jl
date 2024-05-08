#=
    All entropies take base 2

    I'm not checking if something reasonable is fed into the function so use with care...
=#

include("mpartiteLinAlg.jl")
using LinearAlgebra

function ent(rho::AbstractMatrix, csys::Vector{<:Integer}, dims::Vector{<:Real})
    #=
    Conditional von Neumann entropy with conditioning systems indicated by csys
    =#
    rhoB = ptr(rho, csys, dims)
    return entropy(rho) - entropy(rhoB)
end

function m0log(X::AbstractMatrix)
    #=
    Computes matrix log ignoring 0 eigenvalues  
    =#
    F = eigen(X)
    Y = zeros(eltype(X), size(X))
    for i = 1:eachindex(F.values)
        if abs(F.values[i]) > 1e-6
            println(F.values[i])
            println(F.vectors[i, :])
            Y += log(F.values[i]) * F.vectors[i, :] * F.vectors[i, :]'
        end
    end
    return Y
end

function m0log2(X::AbstractMatrix)
    #=
    Computes matrix log ignoring 0 eigenvalues  
    =#
    v, U = eigen(X)
    for i = 1:eachindex(v)
        if abs(v[i]) > 1e-6
            v[i] = log2(v[i])
        end
    end
    return U * Diagonal(v) * U'
end
