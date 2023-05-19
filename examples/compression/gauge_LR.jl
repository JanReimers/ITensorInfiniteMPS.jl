using ITensors, ITensorMPOCompression, ITensorInfiniteMPS
using Revise, Printf

#import ITensorMPOCompression: regform_blocks, extract_blocks,  A0, b0, c0
import ITensorInfiniteMPS: reg_form_iMPO
Base.show(io::IO, f::Float64) = @printf(io, "%1.3e", f) #dumb way to control float output

#H = ΣⱼΣn (½ S⁺ⱼS⁻ⱼ₊n + ½ S⁻ⱼS⁺ⱼ₊n + SᶻⱼSᶻⱼ₊n)

function ITensorInfiniteMPS.unit_cell_terms(::Model"heisenbergNNN"; NNN::Int64)
  opsum = OpSum()
  for n in 1:NNN
      J = 1.0 / n
      opsum += J * 0.5, "S+", 1, "S-", 1 + n
      opsum += J * 0.5, "S-", 1, "S+", 1 + n
      opsum += J, "Sz", 1, "Sz", 1 + n
  end
  return opsum
end

let
    initstate(n) = "↑"
    sites = infsiteinds("S=1/2", 2; initstate, conserve_qns=false)
    ψ = InfMPS(sites, initstate)
    H = InfiniteMPO(Model"heisenbergNNN"(), sites; NNN=5)
    Hc=orthogonalize(H)
    
    @show norm(Hc.G[0]*Hc.AR[1]-H[1]*Hc.G[1])
    @show norm(Hc.G[1]*Hc.AR[2]-H[2]*Hc.G[2])
    #Hc,_=truncate(H)






    
end