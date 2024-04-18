import .MATLAB
"""
    tsirelson_bound(CG::Matrix, scenario::Vector, level::Integer)
    
Upper bounds the Tsirelson bound of a bipartite Bell funcional game `CG`, written in Collins-Gisin notation. `scenario` is vector detailing the number of inputs and outputs, in the order [oa,ob,ia,ib], and `level` is an integer determining the level of the NPA hierarchy.

This function requires Moment to be installed: https://github.com/ajpgarner/moment
It is only available if you first do "import MATLAB" or "using MATLAB"
"""
function tsirelson_bound(CG::Matrix{<:Real}, scenario::Vector{<:Integer}, level::Integer)
    CG = Float64.(CG)
    scenario = Float64.(scenario)
    return MATLAB.mxcall(:mtk_tsirelson, 1, CG, scenario, level)
end
export tsirelson_bound
