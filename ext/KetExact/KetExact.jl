module KetExact

import CyclotomicNumbers as CN
import Ket
import LinearAlgebra as LA

Ket._root_unity(::Type{CN.Cyc{R}}, n::Integer) where {R<:Real} = CN.E(Int64(n))
Ket._sqrt(::Type{CN.Cyc{R}}, n::Integer) where {R<:Real} = CN.root(Int64(n))
Ket._tol(::Type{CN.Cyc{R}}) where {R<:Real} = Base.rtoldefault(R)

LA.norm(v::AbstractVector{CN.Cyc{R}}) where {R<:Real} = Ket._sqrt(CN.Cyc{R}, Int64(sum(abs2, v)))

end # module
