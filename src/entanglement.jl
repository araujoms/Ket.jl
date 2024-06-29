"""
    schmidt_decomposition(ψ::AbstractVector, dims::AbstractVector{<:Integer})
    
Produces the Schmidt decomposition of `ψ` with subsystem dimensions `dims`. Returns the (sorted) Schmidt coefficients λ and isometries U, V such that
kron(U', V')*`ψ` is of Schmidt form.
"""
function schmidt_decomposition(ψ::AbstractVector, dims::AbstractVector{<:Integer})
    length(dims) != 2 && throw(ArgumentError("Two subsystem sizes must be specified."))
    m = transpose(reshape(ψ, dims[2], dims[1])) #necessary because the natural reshaping would be row-major, but Julia does it col-major
    U, λ, V = LA.svd(m)
    return λ, U, conj(V)
end
"""
    schmidt_decomposition(ψ::AbstractVector, dims::AbstractVector{<:Integer})
    
Produces the Schmidt decomposition of `ψ` assuming equally-sized subsystems. Returns the (sorted) Schmidt coefficients λ and isometries U, V such that
kron(U', V')*`ψ` is of Schmidt form.
"""
function schmidt_decomposition(ψ::AbstractVector)
    n = length(ψ)
    d = isqrt(n)
    d^2 != n && throw(ArgumentError("Subsystems are not equally-sized, please specify sizes."))
    return schmidt_decomposition(ψ, [d, d])
end
export schmidt_decomposition
