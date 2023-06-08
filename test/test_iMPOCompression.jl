using ITensors, ITensorMPOCompression, ITensorInfiniteMPS
using Test, Revise

import ITensorMPOCompression: regform_blocks, extract_blocks,  A0, b0, c0

#Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f) #dumb way to control float output

function test_links(Wb1::regform_blocks,Wb2::regform_blocks)
  @test hasinds(Wb1.𝐀̂.W,Wb1.𝐀̂.ileft,Wb1.𝐀̂.iright)
  @test hasinds(Wb1.𝐛̂.W,Wb1.𝐛̂.ileft,Wb1.𝐛̂.iright)
  @test hasinds(Wb1.𝐜̂.W,Wb1.𝐜̂.ileft,Wb1.𝐜̂.iright)
  @test hasinds(Wb1.𝐝̂.W,Wb1.𝐝̂.ileft,Wb1.𝐝̂.iright)
  @test Wb1.𝐝̂.ileft==Wb1.𝐜̂.ileft
  @test Wb1.𝐝̂.iright==Wb1.𝐛̂.iright
  @test Wb1.𝐀̂.ileft==Wb1.𝐛̂.ileft
  @test Wb1.𝐀̂.iright==Wb1.𝐜̂.iright
  @test id(Wb1.𝐀̂.iright)==id(Wb2.𝐀̂.ileft)
  @test dir(Wb1.𝐀̂.iright)==dir(dag(Wb2.𝐀̂.ileft))
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

    Wbs=extract_blocks(H,lr;Abcd=true,fix_inds=true)
    for n in 1:N
      test_links(Wbs[n],Wbs[n+1])
    end
      
  end

  @testset "Truncate/Compress InfiniteCanonicalMPO, H=$(model[1]), qbs=$qns, Ncell=$Ncell, NNN=$NNN" for model in models,
    qns in [false,true], Ncell in [1,3], NNN in [1,4]
      eps= qns ? NNN*1e-14 : NNN*2e-14
      
      initstate(n) = isodd(n) ? "↑" : "↓"
      sites = infsiteinds(model[2], Ncell; initstate, conserve_qns=qns)
      Hi = InfiniteMPO(model[1], sites;NNN=NNN)

      Ho = orthogonalize(Hi) #Use default cutoff, C is non-diagonal
      @test check_ortho(Ho) #AL is left ortho && AR is right ortho
      @test check_gauge_LR(Ho) ≈ 0.0 atol = eps    #ensure GLR[k-1] * AR[k] - AL[k] * GLR[k]
      @test check_gauge_0R(Ho,Hi) ≈ 0.0 atol = eps #ensure G0R[k-1] * AR[k] - H0[k] * G0R[k]

      Ht,BondSpectrums = truncate(Hi) #Use default cutoff,C is now diagonal
      @test check_ortho(Ht) #AL is left ortho && AR is right ortho
      @test check_gauge_LR(Ht) ≈ 0.0 atol = eps    #ensure GLR[k-1] * AR[k] - AL[k] * GLR[k]
      @test check_gauge_0R(Ht,Hi) ≈ 0.0 atol = eps #ensure G0R[k-1] * AR[k] - H0[k] * G0R[k]
      #@show BondSpectrums
  end

  @testset "Try a lattice with alternating S=1/2 and S=1 sites. iMPO. Qns=$qns, Ncell=$Ncell, NNN=$NNN" for qns in [false,true], Ncell in [1,3], NNN in [1,4]
      eps= NNN*2e-14 
      initstate(n) = isodd(n) ? "Dn" : "Up"
      si = infsiteinds(n->isodd(n) ? "S=1" : "S=1/2",Ncell; initstate, conserve_qns=qns)
      Hi = InfiniteMPO(Model"heisenbergNNN"(), si;NNN=NNN)

      Ho = orthogonalize(Hi) #Use default cutoff, C is non-diagonal
      @test check_ortho(Ho) #AL is left ortho && AR is right ortho
      @test check_gauge_LR(Ho) ≈ 0.0 atol = eps    #ensure GLR[k-1] * AR[k] - AL[k] * GLR[k]
      @test check_gauge_0R(Ho,Hi) ≈ 0.0 atol = eps #ensure G0R[k-1] * AR[k] - H0[k] * G0R[k]

      Ht,BondSpectrums = truncate(Hi) #Use default cutoff,C is now diagonal
      @test check_ortho(Ht) #AL is left ortho && AR is right ortho
      @test check_gauge_LR(Ht) ≈ 0.0 atol = eps    #ensure GLR[k-1] * AR[k] - AL[k] * GLR[k]
      @test check_gauge_0R(Ht,Hi) ≈ 0.0 atol = eps #ensure G0R[k-1] * AR[k] - H0[k] * G0R[k]
      #@show BondSpectrums
  end

end

function ITensorInfiniteMPS.unit_cell_terms(::Model"heisenbergNNNs"; NNNs::Vector{Int64})
  opsum = OpSum()
  for i in eachindex(NNNs)
      for n in 1:NNNs[i]
          J = 1.0 / n
          opsum += J * 0.5, "S+", i, "S-", i + n
          opsum += J * 0.5, "S-", i, "S+", i + n
          opsum += J, "Sz", i, "Sz", i + n
      end
  end 
  
  return opsum
end

function delta(s1::Spectrum,s2::Spectrum)
  s1=eigs(s1)
  s2=eigs(s2)
  ds=.√(s1)-.√(s2)
  return sqrt(sum(ds.^2))
end

@testset "Look at rectangular iMPO bond spectra" begin
  initstate(n) = isodd(n) ? "↑" : "↓"
  eps=1e-14
  N=4
  sites = infsiteinds("S=1/2",N; initstate, conserve_qns=true)
  H1234 = InfiniteMPO(Model"heisenbergNNNs"(), sites;NNNs=[1,2,3,4])
  H2341 = InfiniteMPO(Model"heisenbergNNNs"(), sites;NNNs=[2,3,4,1])
  H3412 = InfiniteMPO(Model"heisenbergNNNs"(), sites;NNNs=[3,4,1,2])
  H4123 = InfiniteMPO(Model"heisenbergNNNs"(), sites;NNNs=[4,1,2,3])
  H4321 = InfiniteMPO(Model"heisenbergNNNs"(), sites;NNNs=[4,3,2,1])
  H3214 = InfiniteMPO(Model"heisenbergNNNs"(), sites;NNNs=[3,2,1,4])

  H1234t,BondSpectrumsH1234 = truncate(H1234) #Use default cutoff,C is now diagonal
  @test check_ortho(H1234t) #AL is left ortho && AR is right ortho
  @test check_gauge_LR(H1234t) ≈ 0.0 atol = eps #ensure GLR[k-1] * AR[k] - AL[k] * GLR[k]
  @test check_gauge_0R(H1234t,H1234) ≈ 0.0 atol = 2*eps #ensure G0R[k-1] * AR[k] - H0[k] * G0R[k]
  # @show BondSpectrumsH1234

  H2341t,BondSpectrumsH2341 = truncate(H2341) #Use default cutoff,C is now diagonal
  @test check_ortho(H2341t) #AL is left ortho && AR is right ortho
  @test check_gauge_LR(H2341t) ≈ 0.0 atol = eps #ensure GLR[k-1] * AR[k] - AL[k] * GLR[k]
  @test check_gauge_0R(H2341t,H2341) ≈ 0.0 atol = 2*eps #ensure G0R[k-1] * AR[k] - H0[k] * G0R[k]
  # @show BondSpectrumsH2341

  H3412t,BondSpectrumsH3412 = truncate(H3412) #Use default cutoff,C is now diagonal
  @test check_ortho(H3412t) #AL is left ortho && AR is right ortho
  @test check_gauge_LR(H3412t) ≈ 0.0 atol = eps #ensure GLR[k-1] * AR[k] - AL[k] * GLR[k]
  @test check_gauge_0R(H3412t,H3412) ≈ 0.0 atol = 2*eps #ensure G0R[k-1] * AR[k] - H0[k] * G0R[k]
  # @show BondSpectrumsH3412

  H4123t,BondSpectrumsH4123 = truncate(H4123) #Use default cutoff,C is now diagonal
  @test check_ortho(H4123t) #AL is left ortho && AR is right ortho
  @test check_gauge_LR(H4123t) ≈ 0.0 atol = eps #ensure GLR[k-1] * AR[k] - AL[k] * GLR[k]
  @test check_gauge_0R(H4123t,H4123) ≈ 0.0 atol = 2*eps #ensure G0R[k-1] * AR[k] - H0[k] * G0R[k]
  # @show BondSpectrumsH4123

  H4321t,BondSpectrumsH4321 = truncate(H4321) #Use default cutoff,C is now diagonal
  @test check_ortho(H4321t) #AL is left ortho && AR is right ortho
  @test check_gauge_LR(H4321t) ≈ 0.0 atol = eps #ensure GLR[k-1] * AR[k] - AL[k] * GLR[k]
  @test check_gauge_0R(H4321t,H4321) ≈ 0.0 atol = 2*eps #ensure G0R[k-1] * AR[k] - H0[k] * G0R[k]
  # @show BondSpectrumsH4321

  H3214t,BondSpectrumsH3214 = truncate(H3214) #Use default cutoff,C is now diagonal
  @test check_ortho(H3214t) #AL is left ortho && AR is right ortho
  @test check_gauge_LR(H3214t) ≈ 0.0 atol = eps #ensure GLR[k-1] * AR[k] - AL[k] * GLR[k]
  @test check_gauge_0R(H3214t,H3214) ≈ 0.0 atol = 2*eps #ensure G0R[k-1] * AR[k] - H0[k] * G0R[k]
  # @show BondSpectrumsH3214

  
  @test delta(BondSpectrumsH1234[1],BondSpectrumsH2341[4]) ≈ 0.0 atol = eps
  @test delta(BondSpectrumsH1234[1],BondSpectrumsH3412[3]) ≈ 0.0 atol = eps
  @test delta(BondSpectrumsH1234[1],BondSpectrumsH4123[2]) ≈ 0.0 atol = eps
  @test delta(BondSpectrumsH1234[2],BondSpectrumsH2341[1]) ≈ 0.0 atol = eps
  @test delta(BondSpectrumsH1234[2],BondSpectrumsH3412[4]) ≈ 0.0 atol = eps
  @test delta(BondSpectrumsH1234[2],BondSpectrumsH4123[3]) ≈ 0.0 atol = eps
  @test delta(BondSpectrumsH1234[3],BondSpectrumsH2341[2]) ≈ 0.0 atol = eps
  @test delta(BondSpectrumsH1234[3],BondSpectrumsH3412[1]) ≈ 0.0 atol = eps
  @test delta(BondSpectrumsH1234[3],BondSpectrumsH4123[4]) ≈ 0.0 atol = eps
  @test delta(BondSpectrumsH1234[4],BondSpectrumsH2341[3]) ≈ 0.0 atol = eps
  @test delta(BondSpectrumsH1234[4],BondSpectrumsH3412[2]) ≈ 0.0 atol = eps
  @test delta(BondSpectrumsH1234[4],BondSpectrumsH4123[1]) ≈ 0.0 atol = eps


  _,bs1114 = truncate(InfiniteMPO(Model"heisenbergNNNs"(), sites;NNNs=[1,1,1,4]))
  _,bs1141 = truncate(InfiniteMPO(Model"heisenbergNNNs"(), sites;NNNs=[1,1,4,1]))
  _,bs1411 = truncate(InfiniteMPO(Model"heisenbergNNNs"(), sites;NNNs=[1,4,1,1]))
  _,bs4111 = truncate(InfiniteMPO(Model"heisenbergNNNs"(), sites;NNNs=[4,1,1,1]))
  
  @test delta(bs1114[4],bs1141[3]) ≈ 0.0 atol = eps
  @test delta(bs1114[4],bs1411[2]) ≈ 0.0 atol = eps
  @test delta(bs1114[4],bs4111[1]) ≈ 0.0 atol = eps
  @test delta(bs1114[3],bs1141[2]) ≈ 0.0 atol = eps
  @test delta(bs1114[3],bs1411[1]) ≈ 0.0 atol = eps
  @test delta(bs1114[3],bs4111[4]) ≈ 0.0 atol = eps
  @test delta(bs1114[2],bs1141[1]) ≈ 0.0 atol = eps
  @test delta(bs1114[2],bs1411[4]) ≈ 0.0 atol = eps
  @test delta(bs1114[2],bs4111[3]) ≈ 0.0 atol = eps
  @test delta(bs1114[1],bs1141[4]) ≈ 0.0 atol = eps
  @test delta(bs1114[1],bs1411[3]) ≈ 0.0 atol = eps
  @test delta(bs1114[1],bs4111[2]) ≈ 0.0 atol = eps

end

nothing