module KetMATLAB

import Ket
import MATLAB
import Base.AbstractVecOrTuple

function Ket.tsirelson_bound(CG::Matrix{<:Real}, scenario::AbstractVecOrTuple{<:Integer}, level::Integer)
    CG = Float64.(CG)
    scenario = collect(Float64.(scenario))
    return MATLAB.mxcall(:mtk_tsirelson, 1, CG, scenario, level)
end

end # module
