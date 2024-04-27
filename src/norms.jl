"""
    schatten_norm(X::AbstractMatrix, p::Real)

    Computes Schatten p-norm of matrix X 
"""
function schatten_norm(X::AbstractMatrix, p::Real)
    sv = LA.svdvals(X);
    return LA.norm(sv,p)
end
export schatten_norm

"""
    trace_norm(X::AbstractMatrix)

    Computes trace norm of matrix X 
"""
function trace_norm(X::AbstractMatrix)
    return schatten_norm(X,1)
end
export trace_norm

"""
    frobenius_norm(X::AbstractMatrix)

    Computes frobenius norm of matrix X 
"""
function frobenius_norm(X::AbstractMatrix)
    # Faster than schatten_norm(X,2)
    return sqrt(LA.tr(X * X'))
end
export frobenius_norm

"""
    operator_norm(X::AbstractMatrix)

    Computes operator norm of matrix X 
"""
function operator_norm(X::AbstractMatrix)
    return schatten_norm(X,Inf)
end
export operator_norm

"""
    kyfan_norm(X::AbstractMatrix, k::Int, p::Real = 2)

    Computes Ky-Fan (k,p) norm of matrix X 
"""
function kyfan_norm(X::AbstractMatrix, k::Int, p::Real = 2)
    sv = LA.svdvals(X)
    return LA.norm(sv[1:k],p)
end
export kyfan_norm
