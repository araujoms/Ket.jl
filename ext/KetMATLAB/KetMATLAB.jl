module KetMATLAB

import Ket
import MATLAB

function Ket.tsirelson_bound(CG::Matrix{<:Real}, scenario::Vector{<:Integer}, level::Integer)
    CG = Float64.(CG)
    scenario = Float64.(scenario)
    return MATLAB.mxcall(:mtk_tsirelson, 1, CG, scenario, level)
end

end # module
