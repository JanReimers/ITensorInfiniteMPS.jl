using ITensors, ITensorInfiniteMPS
using Revise

function merge_blocks(H::InfiniteMPO)
  cbs=map(n->combiner(linkind(H,n)), eachindex(H))
  cbs_cv=CelledVector{ITensor}(cbs,translator(H))
  return InfiniteMPO(map(n->dag(cbs_cv[n-1])*H[n]*cbs_cv[n],eachindex(H)))
end

spaces(H::InfiniteMPO)=map(n->length(space(linkind(H,n))),eachindex(H))

include("../../test/hamiltonians/fqhe.jl")

let 
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

  @show get_Dw(Hi) spaces(Hi)

  Hc=merge_blocks(Hi)
  @show get_Dw(Hc) spaces(Hc)

  Ho=orthogonalize(Hi)
  @show get_Dw(Ho.AL) 
  @show get_Dw(Ho.AR) 
  @assert translator(Hi)==translator(Ho.AR)
  @assert translator(Hi)==translator(Ho.AL)
  @assert translator(Hi)==translator(Ho.H0)
  

  Ht,BondSpectrums = ITensors.truncate(Hi) #Warning about interferance with Base.truncate(file,n)
  @show get_Dw(Ht.AL) 
  @show get_Dw(Ht.AR) 
  @show BondSpectrums
  @assert translator(Hi)==translator(Ht.AR)
  @assert translator(Hi)==translator(Ht.AL)
  @assert translator(Hi)==translator(Ht.H0)

  vumps_kwargs = (
        multisite_update_alg="parallel",
        tol=1e-5,
        maxiter=50,
        outputlevel=1,
        return_e=true,
        time_step=-Inf,
      )

  ψ,(eᴸ, eᴿ) = tdvp(Ht, ψ; vumps_kwargs...)
  for _ in 1:1
      ψ = subspace_expansion(ψ, Ht; cutoff=1e-8,maxdim=32)
      ψ,(eᴸ, eᴿ) = tdvp(Ht, ψ; vumps_kwargs...)      
  end
  # Ht,BondSpectrums = ITensors.truncate(Hi;cutoff=1e-10)
  # @show get_Dw(Ht.AL) spaces(Ht.AL)
  # @show BondSpectrums

end # let
nothing
