using ITensors, ITensorMPOCompression, ITensorInfiniteMPS
using Test, Revise

import ITensorMPOCompression: check, regform_blocks, extract_blocks, A0, b0, c0, vector_o2, MPO, get_Dw
import ITensorMPOCompression: check_ortho
import ITensorInfiniteMPS: check_gauge
#Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f) #dumb way to control float output

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

function test_links(Wb1::regform_blocks,Wb2::regform_blocks)
  @test hasinds(Wb1.𝐀̂,Wb1.irA,Wb1.icA)
  @test hasinds(Wb1.𝐛̂,Wb1.irb,Wb1.icb)
  @test hasinds(Wb1.𝐜̂,Wb1.irc,Wb1.icc)
  @test hasinds(Wb1.𝐝̂,Wb1.ird,Wb1.icd)
  @test Wb1.ird==Wb1.irc
  @test Wb1.icd==Wb1.icb
  @test Wb1.irA==Wb1.irb
  @test Wb1.icA==Wb1.icc
  @test id(Wb1.icA)==id(Wb2.irA)
  @test dir(Wb1.icA)==dir(dag(Wb2.irA))
end

function verify_links(H::InfiniteMPO)
  Ncell = length(H)
  @test order(H[1]) == 4
  @test hastags(H[1], "Link,c=0,l=$Ncell")
  @test hastags(H[1], "Link,c=1,l=1")
  @test hastags(H[1], "Site,c=1,n=1")
  for n in 2:Ncell
    @test order(H[n]) == 4
    @test hastags(H[n], "Link,c=1,l=$(n-1)")
    @test hastags(H[n], "Link,c=1,l=$n")
    @test hastags(H[n], "Site,c=1,n=$n")
    il, = inds(H[n - 1]; tags="Link,c=1,l=$(n-1)")
    ir, = inds(H[n]; tags="Link,c=1,l=$(n-1)")
    @test id(il) == id(ir)
  end
  il, = inds(H[1]; tags="Link,c=0,l=$Ncell")
  ir, = inds(H[Ncell]; tags="Link,c=1,l=$Ncell")
  @test id(il) == id(ir)
  @test order(H[Ncell]) == 4
end

models = [(Model"heisenbergNNN"(), "S=1/2"), (Model"hubbardNNN"(), "Electron")]

@testset "InfinitMPO compression tests" begin
  @testset "Production of iMPOs from AutoMPO, N=$N, NNN=$NNN, qns=$qns" for model in models, 
    qns in [false, true],
    N in [1, 2, 3, 4],
    NNN in [1, 2, 3, 4, 5]

    initstate(n) = "↑"
    sites = infsiteinds(model[2], N; initstate, conserve_qns=qns)
    H = InfiniteMPO(model[1], sites; NNN=NNN)
    @test length(H) == N
    Dws = get_Dw(H)
    @test all(y -> y == Dws[1], Dws)
    verify_links(H)
  end

  @testset "Extract blocks iMPO N=$N, qns=$qns with fix_inds=true ul=$ul" for model in models,
    N in [1,4], 
    NNN in [1,4],
    qns in [false,true], 
    ul in [lower]
    
    initstate(n) = "↑"
    sites = infsiteinds(model[2], N;initstate, conserve_qns=qns)
    H = ITensorInfiniteMPS.reg_form_iMPO(InfiniteMPO(model[1], sites; NNN=NNN))
    lr = ul == lower ? left : right

    Wbs=extract_blocks(H,lr;fix_inds=true)
    for n in 1:N-1
      test_links(Wbs[n],Wbs[n+1])
    end
    test_links(Wbs[N],Wbs[1])
      
  end

  # @testset "Convert upper iMPO to lower H=$(model[1]), qns=$qns" for model in models, qns in [false,true], N in [1,2,3,4], NNN in [1,4]
  #   initstate(n) = "↑"
  #   sites = infsiteinds(model[2], N; initstate, conserve_qns=false)
  #   Hu = reg_form_iMPO(model[1](sites, NNN;ul=upper);honour_upper=true)
  #   @test is_regular_form(Hu)
  #   @test Hu.ul==upper

  #   Hl = transpose(Hu)
  #   @test Hu.ul==upper
  #   @test is_regular_form(Hu)
  #   @test Hl.ul==lower
  #   @test is_regular_form(Hl;verbose=true)
  #   @test !check_ortho(Hu,left)
  #   @test !check_ortho(Hl,left)
  #   @test !check_ortho(Hu,right)
  #   @test !check_ortho(Hl,right)
    
  #   ac_orthogonalize!(Hl,left)
  #   @test Hl.ul==lower
  #   @test is_regular_form(Hl;verbose=true)
  #   @test check_ortho(Hl,left)
  #   @test !check_ortho(Hu,left)
  #   @test !check_ortho(Hu,right)
  #   Hu1=transpose(Hl)
  #   @test Hu1.ul==upper
  #   @test is_regular_form(Hu1)
  #   @test check_ortho(Hu1,left;verbose=true)
    
  # end


  @testset "Truncate/Compress InfiniteCanonicalMPO, H=$(model[1]), qbs=$qns, Ncell=$Ncell, NNN=$NNN" for model in models,
    qns in [false,true], Ncell in [1,3], NNN in [1,4]
      eps=NNN*1e-14
      initstate(n) = isodd(n) ? "↑" : "↓"
      sites = infsiteinds(model[2], Ncell; initstate, conserve_qns=qns)
      Hi = InfiniteMPO(model[1], sites;NNN=NNN)

      Ho::InfiniteCanonicalMPO = orthogonalize(Hi) #Use default cutoff, C is non-diagonal
      @test check_ortho(Ho) #AL is left ortho && AR is right ortho
      @test check_gauge(Ho) ≈ 0.0 atol = eps #ensure C[n - 1] * AR[n] - AL[n] * C[n]

      Ht,BondSpectrums = truncate(Hi) #Use default cutoff,C is now diagonal
      @test check_ortho(Ht) #AL is left ortho && AR is right ortho
      @test check_gauge(Ht) ≈ 0.0 atol = eps #ensure C[n - 1] * AR[n] - AL[n] * C[n]
      #@show BondSpectrums
  end

  @testset "Try a lattice with alternating S=1/2 and S=1 sites. iMPO. Qns=$qns, Ncell=$Ncell, NNN=$NNN" for qns in [false,true], Ncell in [1,3], NNN in [1,4]
      eps=NNN*1e-14
      initstate(n) = isodd(n) ? "Dn" : "Up"
      si = infsiteinds(n->isodd(n) ? "S=1" : "S=1/2",Ncell; initstate, conserve_qns=qns)
      Hi = InfiniteMPO(Model"heisenbergNNN"(), si;NNN=NNN)

      Ho::InfiniteCanonicalMPO = orthogonalize(Hi) #Use default cutoff, C is non-diagonal
      @test check_ortho(Ho) #AL is left ortho && AR is right ortho
      @test check_gauge(Ho) ≈ 0.0 atol = eps #ensure C[n - 1] * AR[n] - AL[n] * C[n]

      Ht,BondSpectrums = truncate(Hi) #Use default cutoff,C is now diagonal
      @test check_ortho(Ht) #AL is left ortho && AR is right ortho
      @test check_gauge(Ht) ≈ 0.0 atol = eps #ensure C[n - 1] * AR[n] - AL[n] * C[n]
      @show BondSpectrums
  end

end
nothing