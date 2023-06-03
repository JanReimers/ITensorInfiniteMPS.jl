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
      #@show BondSpectrums
  end

end
nothing