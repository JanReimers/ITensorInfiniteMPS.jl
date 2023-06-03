using ITensors, ITensorMPOCompression, ITensorInfiniteMPS
using Revise

import ITensorInfiniteMPS: reg_form_iMPO, gauge_fix!
#  
#     NNN = Number of Nearest Neightbours
#
function ITensorInfiniteMPS.unit_cell_terms(::Model"heisenbergNNNs"; NNNs::Vector{Int64})
    opsum = OpSum()
    for i in eachindex(NNNs)
        for n in 1:NNNs[i]
            J = i / n^2
            opsum += J * 0.5, "S+", i, "S-", i + n
            opsum += J * 0.5, "S-", i, "S+", i + n
            opsum += J, "Sz", i, "Sz", i + n
        end
    end 
    
    return opsum
end

initstate(n) = "↑"
N=4
sites = infsiteinds("S=1/2", N; initstate, conserve_qns=false)
H = InfiniteMPO(Model"heisenbergNNNs"(), sites; NNNs=[2,2,2,1])
# pprint(H[1])
# pprint(H[2])
@show get_Dw(H)
# Hrf=reg_form_iMPO(H)
# gauge_fix!(Hrf)
# @show get_Dw(Hrf)
# pprint(Hrf[1])
# pprint(Hrf[2])
Hc=orthogonalize(H)
# pprint(Hc.AL[1])
# pprint(Hc.AL[2])
@show get_Dw(Hc.AL)
