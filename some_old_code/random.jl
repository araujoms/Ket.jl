#=
Generating random objects
=#
using LinearAlgebra
using BenchmarkTools
using Random

function randRho(d::Integer, rank::Integer = d)
    #=
    Generates random dxd density matrix with randk 
    Parameters: 
        d   -   dimension
        rank -  rank of final matrix 
    Returns:
        rho   -   Matrix{ComplexF64}
    =#

    # Create the spectrum of the density matrix
    spectrum = rand(rank)
    spectrum .= spectrum ./ sum(spectrum)
    spectrum = [spectrum; zeros(d - rank)]

    # Apply a Haar random unitary 
    U = random_unitary(d)
    rho = U * Diagonal(spectrum) * U'

    return Hermitian(rho)
end

function randRhoR(d::Integer, rank::Integer = d)
    #=
    Generates random real dxd density matrix with given rank 
    Parameters: 
        d   -   dimension
        rank -  rank of final matrix 
    Returns:
        rho   -   Matrix{Float64}
    =#

    # Create the spectrum of the density matrix
    spectrum = rand(rank)
    spectrum .= spectrum ./ sum(spectrum)
    spectrum = [spectrum; zeros(d - rank)]

    # Apply random orthogonal matrix
    U = randUnitaryR(d)
    rho = U * Diagonal(spectrum) * U'

    return Symmetric(rho)
end

function randIsometry(d1::Integer, d2::Integer)
    #=
    Generates random isometry from a d1 dimensional 
    Hilbert space to a d2 dimensional Hilbert space 
        d1   -  dimension of input space 
        d2   -  dimension of output space
    Returns:
        V   -   Isometry

    TODO: Not checked carefully that this method produces a "good" sample.
          Constructs isometry via embedding + unitary transformation. 
    =#

    embedding = [I(d1); zeros((d2 - d1), d1)]
    U = randUnitary(d2)

    return U * embedding
end

function randPOVM(d::Integer, nout::Integer)
    #=
    Generates a random POVM with nout outcomes on a d dimensional space 
        nout   -  number of outcomes 
        d      -  dimension of space 
    Returns:
        M   -   POVM Vector{Matrix{ComplexF64}}

    Uses definition from https://arxiv.org/abs/1902.04751 with n=d
    =#
    M = [Matrix{ComplexF64}(undef, d, d) for i = 1:nout]

    V = randIsometry(d, nout * d)
    for i = 1:nout
        proj = Diagonal(I(nout)[i, :])
        M[i] = V' * kron(proj, I(d)) * V
    end
    return M
end
