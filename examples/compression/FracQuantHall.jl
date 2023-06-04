using ITensors
using ITensorMPOCompression
using ITensorInfiniteMPS
using Revise

import ITensorInfiniteMPS: reg_form_iMPO
import ITensorMPOCompression: gauge_fix!, is_gauge_fixed

include("../../test/hamiltonians/fqhe.jl")


ψ = InfMPS(s, initstate);

Hm = InfiniteMPOMatrix(fqhe_model, s; fqhe_model_params...);
for H in Hm
  lx,ly=size(H)
  for ix in 1:lx
    for iy in 1:ly
      M=H[ix,iy]
      if !isempty(M)
        ITensors.checkflux(M)
      end
    end
  end
end

Hi=InfiniteMPO(Hm)

ITensors.checkflux.(Hi)

@show get_Dw(Hi)
Hc=orthogonalize(Hi)
@show get_Dw(Hc.AL)
Ht,BondSpectrums = truncate(Hi)
@show get_Dw(Ht.AL)
@show BondSpectrums
Ht,BondSpectrums = truncate(Hi;cutoff=1e-10)
@show get_Dw(Ht.AL)
@show BondSpectrums

nothing
