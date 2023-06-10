using ITensors, ITensorMPOCompression, ITensorInfiniteMPS
using Revise, Printf,Test

import Base: inv
import ITensorInfiniteMPS: reg_form_iMPO, left_environment, right_environment, AbstractInfiniteMPS
Base.show(io::IO, f::Float64) = @printf(io, "%1.6f", f) #dumb way to control float output

#   H = ΣⱼΣₙ (½ S⁺ⱼS⁻ⱼ₊ₙ + ½ S⁻ⱼS⁺ⱼ₊ₙ + SᶻⱼSᶻⱼ₊ₙ)

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
function ITensorInfiniteMPS.unit_cell_terms(::Model"hubbardNNN"; NNN::Int64)
  U::Float64 = 0.25
  t::Float64 = 1.0
  V::Float64 = 0.5
  opsum = OpSum()
  opsum += (U, "Nupdn", 1)
  for n in 1:NNN
    tj, Vj = t / n, V / n
    opsum += -tj, "Cdagup", 1, "Cup", 1 + n
    opsum += -tj, "Cdagup", 1 + n, "Cup", 1
    opsum += -tj, "Cdagdn", 1, "Cdn", 1 + n
    opsum += -tj, "Cdagdn", 1 + n, "Cdn", 1
    opsum += Vj, "Ntot", 1, "Ntot", 1 + n
  end
  return opsum
end

models = [
  (Model"heisenbergNNN"(), "S=1/2"), 
  (Model"hubbardNNN"(), "Electron")
  ]

@testset "Gauge transforms and inversions, H=$(model[1]), N=$N, NNN=$NNN, qns=$qns" for model in models, N in 1:4, NNN in [1,2,5,10], qns in [false,true]
  eps=3e-14
  initstate(n) = "↑"
  sites = infsiteinds(model[2], N; initstate, conserve_qns=qns)
  ψ = InfMPS(sites, initstate)
  H = InfiniteMPO(model[1], sites; NNN=NNN)
  Ho=orthogonalize(H)
  @test check_gauge_LR(Ho) ≈ 0.0 atol = eps*NNN*N
  @test check_gauge_0R(Ho,H) ≈ 0.0 atol = eps*NNN*N
  
  Ht,ss=truncate(H)
  @test check_gauge_LR(Ht) ≈ 0.0 atol = eps*NNN*N
  @test check_gauge_0R(Ht,H) ≈ 0.0 atol = eps*NNN*N
end

@testset "Reduced L/R environments, H=$(model[1]), N=$N, NNN=$NNN, qns=$qns" for model in models, N in 1:4, NNN in [1,2,5,10], qns in [false,true]
  eps=3e-14
  initstate(n) = "↑"
  sites = infsiteinds(model[2], N; initstate, conserve_qns=qns)
  ψ = InfMPS(sites, initstate)
  H = InfiniteMPO(model[1], sites; NNN=NNN)
  Ht,ss=truncate(H)
  L,el=left_environment(H,ψ)
  R,er=right_environment(H,ψ)
  #
  #  Calculate the envirments for the (non-triangular) compressed Hamiltonian
  #
  Lc=CelledVector{ITensor}(undef,N)
  Rc=CelledVector{ITensor}(undef,N)
  for k in 1:N
    Lc[k]=L[k]*Ht.G0R[k]
    Rc[k]=R[k]*inv(Ht.G0R[k])
    @test order(Lc[k])==3
    @test order(Rc[k])==3
  end
  #
  #  Check that they are interlaced pseudo-eigenvectors
  #
  for k in 1:N
    Lck=Lc[k-1]*Ht.AR[k]*ψ.AL[k]*dag(ψ.AL[k]')
    Rck=Rc[k+1]*Ht.AR[k+1]*ψ.AR[k+1]*dag(ψ.AR[k+1]')
    if k==1
      il,ir=ITensorMPOCompression.parse_links(Ht.AR[k])
      il1,ir1=ITensorMPOCompression.parse_links(ψ.AL[k])
      il2,ir2=ITensorMPOCompression.parse_links(ψ.AR[k+1])
      Lck[ir=>1,ir1=>1,ir1'=>1]-=el
      Rck[ir=>dim(ir),il2=>1,il2'=>1]-=er
    end

    # @show Rck-Rc[k] er
    @test norm(Lck-Lc[k]) ≈ 0.0 atol = eps*NNN*N
    @test norm(Rck-Rc[k]) ≈ 0.0 atol = eps*NNN*N
  end


end

#
#  Benchmark tests
#
# using BenchmarkTools

# global Dglobal=0

# function runvumps(ψ::AbstractInfiniteMPS,H,vumps_kwargs,maxdim::Int64)
#   ψ = tdvp(H, ψ; vumps_kwargs...)
#   for nex=1:4
#     ψ = subspace_expansion(ψ, H; cutoff=1e-8, maxdim)
#     ψ = tdvp(H, ψ; vumps_kwargs...)  
#     global Dglobal=dim(commoninds(ψ.AL[1], ψ.C[1]))   
#     if Dglobal>=maxdim
#       break
#     end
#   end 
# end

# function InfiniteCanonicalMPO_orth(model::Model, s::CelledVector; kwargs...)
#   Hi=InfiniteMPO(model,s;kwargs...)
#   Hc=orthogonalize(Hi)
#   return Hc
# end

# function InfiniteCanonicalMPO(model::Model, s::CelledVector; kwargs...)
#   Hi=InfiniteMPO(model,s;kwargs...)
#   Hc,ss=truncate(Hi)
#   return Hc
# end

# function ITensorMPOCompression.get_Dw(H::InfiniteMPOMatrix)
#   return get_Dw(InfiniteMPO(H))
# end

# Htypes= [InfiniteMPOMatrix,InfiniteMPO,InfiniteCanonicalMPO_orth,InfiniteCanonicalMPO]
# let
#   println("    Model         Hamiltonian Type               maxD   D  Dw   NNN   Time(sec)")
# for NNN in [1,3,5], maxD in [2,4], model in models, Htype in Htypes
#   vumps_kwargs = (
#       multisite_update_alg="sequential",
#       tol=1e-8,
#       maxiter=5,
#       outputlevel=0,
#       time_step=-Inf,
#     )

#   # initstate(n) = "↑"
#   initstate(n) = isodd(n) ? "↑" : "↓"
#   sites = infsiteinds(model[2], 2; initstate, conserve_qns=true)
#   ψ = InfMPS(sites, initstate)
#   H = Htype(model[1], sites;NNN=NNN)
#   Dw=get_Dw(H)[1]
#   t = @benchmark runvumps($ψ,$H,$vumps_kwargs,$maxD) samples=1 evals=1
#   t=mean(t).time*1e-9
#   #println("Model=$model[1], H=$Htype, Dw=$Dw, NNN=$NNN, t=$t (sec)")
#   ms=rpad(string(model[1])[8:end-3],16," ")
#   hs=rpad(Htype,30," ")
#   @printf("%s %s %4i %4i %4i %4i    %1.3f\n",ms, hs ,maxD, Dglobal, Dw,NNN, t)
# end
# end
