module KetExact

import CyclotomicNumbers as CN
import Ket

Ket._root_unity(::Type{CN.Cyc{R}}, n::Integer) where {R<:Real} = CN.E(Int64(n))
Ket._sqrt(::Type{CN.Cyc{R}}, n::Integer) where {R<:Real} = CN.root(Int64(n))
Ket._tol(::Type{CN.Cyc{R}}) where {R<:Real} = Base.rtoldefault(R)

end # module
