using ITensors, ITensorMPOCompression, ITensorInfiniteMPS
using Revise, Printf,Test

import Base: inv
import ITensorInfiniteMPS: reg_form_iMPO, left_environment, right_environment
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
  (Model"hubbardNNN"(), "Electron")
  ]


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
    Lc[k]=L[k]*Ht.G[k]
    Rc[k]=R[k]*inv(Ht.G[k])
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

# let
#     initstate(n) = "↑"
#     sites = infsiteinds("Electron", 2; initstate, conserve_qns=false)
#     ψ = InfMPS(sites, initstate)
#     H = InfiniteMPO(Model"hubbardNNN"(), sites; NNN=10)
#     Ho=orthogonalize(H)
#     @show check_gauge(Ho)
#     @show check_gauge(Ho,H)
    
#     Ht,ss=truncate(H)
#     @show check_gauge(Ht)
#     @show check_gauge(Ht,H)
#     L,el=left_environment(H,ψ)
#     R,er=right_environment(H,ψ)
#     LT=L[0]*H[1]*ψ.AL[1]*dag(ψ.AL[1]')
#     RT=R[2]*H[2]*ψ.AR[2]*dag(ψ.AR[2]')
#     # @show inds(LT) inds(L[1]) LT-L[1] el
#     # @show inds(RT) inds(R[1]) RT-R[1] er

#     #
#     #  Evaluate L_comp(k)_comp=L(k)*G(K)
#     #
#     N=nsites(H)
#     Lc=CelledVector{ITensor}(undef,N)
#     Rc=CelledVector{ITensor}(undef,N)
#     for k in 1:N
#       Lc[k]=L[k]*Ht.G[k]
#       Rc[k]=R[k]*Ht.G[k]
#       @assert order(Rc[k])==3
#     end
#     LT=Lc[0]*Ht.AR[1]*ψ.AL[1]*dag(ψ.AL[1]')
#     RT=Rc[2]*Ht.AR[2]*ψ.AR[2]*dag(ψ.AR[2]')
#     @assert order(LT)==3
#     @assert order(RT)==3
#     il,ir=ITensorMPOCompression.parse_links(Ht.AR[1])
#     il1,ir1=ITensorMPOCompression.parse_links(ψ.AL[1])
#     il2,ir2=ITensorMPOCompression.parse_links(ψ.AR[2])
#     LT[ir=>1,ir1=>1,ir1'=>1]-=el
#     RT[ir=>dim(ir),il2=>1,il2'=>1]-=er

#     @show norm(LT-Lc[1])
#     @show norm(RT-Rc[1])


# end
nothing