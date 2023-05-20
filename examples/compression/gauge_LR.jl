using ITensors, ITensorMPOCompression, ITensorInfiniteMPS
using Revise, Printf,Test

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
  (Model"hubbardNNN"(), "Electron")]


@testset "Gauge transforms and inversions, H=$(model[1]), N=$N, NNN=$NNN, qns=$qns" for model in models, N in 1:4, NNN in [1,2,5,10], qns in [false,true]
  eps=3e-14
  initstate(n) = "↑"
  sites = infsiteinds(model[2], N; initstate, conserve_qns=qns)
  ψ = InfMPS(sites, initstate)
  H = InfiniteMPO(model[1], sites; NNN=NNN)
  Ho=orthogonalize(H)
  @test check_gauge(Ho) ≈ 0.0 atol = eps*NNN*N
  @test check_gauge(Ho,H) ≈ 0.0 atol = eps*NNN*N
  
  Ht,ss=truncate(H)
  @test check_gauge(Ht) ≈ 0.0 atol = eps*NNN*N
  @test check_gauge(Ht,H) ≈ 0.0 atol = eps*NNN*N
end

# let
#     initstate(n) = "↑"
#     sites = infsiteinds("Electron", 1; initstate, conserve_qns=false)
#     ψ = InfMPS(sites, initstate)
#     H = InfiniteMPO(Model"hubbardNNN"(), sites; NNN=10)
#     Ho=orthogonalize(H)
#     @show check_gauge(Ho)
#     @show check_gauge(Ho,H)
    
#     Ht,ss=truncate(H)
#     @show check_gauge(Ht)
#     @show check_gauge(Ht,H)
    
# end
nothing