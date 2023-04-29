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
  
  @testset "Extract blocks iMPO N=$N, qns=$qns with fix_inds=true ul=$ul" for N in [1,4], qns in [false,true], ul in [lower]
    initstate(n) = "↑"
    NNN = 2 #Include 2nd nearest neighbour interactions
    sites = infsiteinds("Electron", N;initstate, conserve_qns=qns)
    H = reg_form_iMPO(Hubbard_AutoiMPO(sites, NNN; ul=ul);honour_upper=true)
  
    lr = ul == lower ? left : right
  
    Wbs=extract_blocks(H,lr;fix_inds=true)
    for n in 1:N-1
      test_links(Wbs[n],Wbs[n+1])
    end
    test_links(Wbs[N],Wbs[1])
     
  end
  
  models = [
  (transIsing_iMPO, "S=1/2", true),
  (transIsing_AutoiMPO, "S=1/2", true),
  (Heisenberg_AutoiMPO, "S=1/2", true),
  (Heisenberg_AutoiMPO, "S=1", true),
  (Hubbard_AutoiMPO, "Electron", false),
]

import ITensorMPOCompression: check, extract_blocks, A0, b0, c0, vector_o2, MPO

@testset "Gauge fix infinite $(model[1]), N=$N, NNN=$NNN, qns=$qns, ul=$ul" for model in models,
  qns in [false,true],
  ul in [lower,upper],
  N in [1,3],
  NNN in [1,4]

  eps = 1e-14
  initstate(n) = "↑"
  si = infsiteinds(model[2], N; initstate, conserve_qns=qns)
  # ψ = InfMPS(si, initstate)
  # for n in 1:N
  #     ψ[n] = randomITensor(inds(ψ[n]))
  # end

  H0 = model[1](si, NNN; ul=ul)
  Hrf = reg_form_iMPO(H0)
  pre_fixed = model[3] #Hamiltonian starts gauge fixed

  # Hsum0=InfiniteSum{MPO}(InfiniteMPO(Hrf),NNN)
  # E0=expect(ψ,Hsum0)

  @test pre_fixed == is_gauge_fixed(Hrf; eps=eps)
  @test is_regular_form(Hrf)
  gauge_fix!(Hrf)
  Wb = extract_blocks(Hrf[1], left; all=true)
  @test norm(b0(Wb)) < eps
  @test norm(c0(Wb)) < eps
  @test is_gauge_fixed(Hrf; eps=eps)
  @test is_regular_form(Hrf)

  # Hsum1=InfiniteSum{MPO}(InfiniteMPO(Hrf),NNN)
  # E1=expect(ψ,Hsum1)
  # @show E0 E1

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

@testset "Production of iMPOs from AutoMPO, Ncell=$Ncell, NNN=$NNN, qns=$qns" for qns in
                                                                                [
    false, true
    ],
    Ncell in [1, 2, 3, 4],
    NNN in [1, 2, 3, 4, 5]

    initstate(n) = "↑"
    site_type = "S=1/2"
    si = infsiteinds(site_type, Ncell; initstate, conserve_qns=qns)
    H = transIsing_AutoiMPO(si, NNN; ul=lower)
    @test length(H) == Ncell
    Dws = get_Dw(H)
    @test all(y -> y == Dws[1], Dws)
    verify_links(H)
end

makeHs = [
    (transIsing_iMPO, "S=1/2"),
    (transIsing_AutoiMPO, "S=1/2"),
    (Heisenberg_AutoiMPO, "S=1/2"),
    (Hubbard_AutoiMPO, "Electron"),
  ]

  @testset "Reg for H=$(makeH[1]), ul=$ul, qns=$qns" for makeH in makeHs,
    qns in [false, true],
    ul in [lower, upper],
    Ncell in 1:5

    initstate(n) = "↑"
    sites = infsiteinds(makeH[2], Ncell; initstate, conserve_qns=qns)
    H = makeH[1](sites, 2; ul=ul)
    @test is_regular_form(H, ul)
    @test hasqns(H[1]) == qns
    verify_links(H)
  end

  models = [
    (transIsing_iMPO, "S=1/2", true),
    (transIsing_AutoiMPO, "S=1/2", true),
    (Heisenberg_AutoiMPO, "S=1/2", true),
    (Heisenberg_AutoiMPO, "S=1", true),
    (Hubbard_AutoiMPO, "Electron", false),
  ]

  @testset "Convert upper iMPO to lower H=$(model[1]), qns=$qns" for model in models, qns in [false,true], N in [1,2,3,4], NNN in [1,4]
    initstate(n) = "↑"
    sites = infsiteinds(model[2], N; initstate, conserve_qns=false)
    Hu = reg_form_iMPO(model[1](sites, NNN;ul=upper);honour_upper=true)
    @test is_regular_form(Hu)
    @test Hu.ul==upper

    Hl = transpose(Hu)
    @test Hu.ul==upper
    @test is_regular_form(Hu)
    @test Hl.ul==lower
    @test is_regular_form(Hl;verbose=true)
    @test !check_ortho(Hu,left)
    @test !check_ortho(Hl,left)
    @test !check_ortho(Hu,right)
    @test !check_ortho(Hl,right)
    
    ac_orthogonalize!(Hl,left)
    @test Hl.ul==lower
    @test is_regular_form(Hl;verbose=true)
    @test check_ortho(Hl,left)
    @test !check_ortho(Hu,left)
    @test !check_ortho(Hu,right)
    Hu1=transpose(Hl)
    @test Hu1.ul==upper
    @test is_regular_form(Hu1)
    @test check_ortho(Hu1,left;verbose=true)
    
  end

  models = [
    (transIsing_iMPO, "S=1/2"),
    (transIsing_AutoiMPO, "S=1/2"),
    (Heisenberg_AutoiMPO, "S=1/2"),
    (Heisenberg_AutoiMPO, "S=1"),
    (Hubbard_AutoiMPO, "Electron"),
  ]

  #
  #  This now gets test in infinite_canonical_mpo.jl
  #
  # @testset "Orthogonalize iMPO Check gauge relations, H=$(model[1]), ul=$ul, qbs=$qns, N=$N, NNN=$NNN" for model in
  #                                                                                                          models,
  #   ul in [lower,upper],
  #   qns in [false, true],
  #   N in [1, 3],
  #   NNN in [1, 4]

  #   eps = NNN * 1e-14
  #   initstate(n) = "↑"
  #   si = infsiteinds(model[2], N; initstate, conserve_qns=qns)
  #   H0 = reg_form_iMPO(model[1](si, NNN; ul=ul))
  #   HL = copy(H0)
  #   @test is_regular_form(HL)
  #   GL = ac_orthogonalize!(HL, left; verbose=verbose1)
  #   DwL = Base.max(get_Dw(HL)...)
  #   @test is_regular_form(HL)
  #   @test check_ortho(HL, left) #expensive does V_dagger*V=Id
  #   for n in 1:N
  #     @test norm(HL[n].W * GL[n] - GL[n - 1] * H0[n].W) ≈ 0.0 atol = eps
  #   end

  #   HR = copy(H0)
  #   GR = ac_orthogonalize!(HR, right; verbose=verbose1)
  #   DwR = Base.max(get_Dw(HR)...)
  #   @test is_regular_form(HR)
  #   @test check_ortho(HR, right) #expensive does V_dagger*V=Id
  #   for n in 1:N
  #     @test norm(GR[n - 1] * HR[n].W - H0[n].W * GR[n]) ≈ 0.0 atol = eps
  #   end
  #   HR1 = copy(HL)
  #   G = ac_orthogonalize!(HR1, right; verbose=verbose1)
  #   DwLR = Base.max(get_Dw(HR1)...)
  #   @test is_regular_form(HR1)
  #   @test check_ortho(HR1, right) #expensive does V_dagger*V=Id
  #   for n in 1:N
  #     # D1=G[n-1]*HR1[n].W
  #     # @assert order(D1)==4
  #     # D2=HL[n].W*G[n]
  #     # @assert order(D2)==4
  #     @test norm(G[n - 1] * HR1[n].W - HL[n].W * G[n]) ≈ 0.0 atol = eps
  #   end
  # end

  models = [
    (transIsing_iMPO, "S=1/2"),
    (transIsing_AutoiMPO, "S=1/2"),
    (Heisenberg_AutoiMPO, "S=1/2"),
    (Heisenberg_AutoiMPO, "S=1"),
    (Hubbard_AutoiMPO, "Electron"),
  ]

  #
  #  This now gets test in infinite_canonical_mpo.jl
  #
  @testset "Truncate/Compress iMPO Check gauge relations, H=$(model[1]), ul=$ul, qbs=$qns, N=$N, NNN=$NNN" for model in
                                                                                                               models,
    ul in [lower,upper],
    qns in [false,true],
    N in [1,2],
    NNN in [1,4]

    initstate(n) = "↑"
    makeH = model[1]
    site_type = model[2]
    eps = qns ? 1e-14 * NNN : 3e-14 * NNN #dense and larger NNN both get more roundoff noise.
    si = infsiteinds(site_type, N; initstate, conserve_qns=qns)
    H0 = reg_form_iMPO(model[1](si, NNN; ul=ul))
    @test is_regular_form(H0)
    Dw0 = Base.max(get_Dw(H0)...)
    #
    #  Do truncate outputting left ortho Hamiltonian
    #
    HL, HR, Ss, ss  = truncate!(H0; verbose=verbose1)
    #@show Ss ss
    @test typeof(storage(Ss[1])) == (
      if qns
        NDTensors.DiagBlockSparse{Float64,Vector{Float64},2}
      else
        Diag{Float64,Vector{Float64}}
      end
    )

    DwL = Base.max(get_Dw(HL)...)
    @test is_regular_form(HL)
    @test check_ortho(HL, left)
    @test check_ortho(HR, right)
    #
    #  Now test guage relations using the diagonal singular value matrices
    #  as the gauge transforms.
    #
    for n in 1:N
      # @show inds(Ss[n-1]) inds(HR[n].W,tags="Link") inds(Ss[n]) inds(HL[n].W,tags="Link") 
      D1 = Ss[n - 1] * HR[n].W
      @assert order(D1) == 4
      D2 = HL[n].W * Ss[n]
      @assert order(D2) == 4
      @test norm(Ss[n - 1] * HR[n].W - HL[n].W * Ss[n]) ≈ 0.0 atol = eps
    end
    if verbose
      @printf " %4i %4i   %4i   %4i  %4i \n" N NNN Dw0 DwL DwR
    end
  end

  import ITensorMPOCompression: check_gauge

Base.show(io::IO, f::Float64) = @printf(io, "%1.3f", f) #dumb way to control float output

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
    (Model"heisenbergNNN", "S=1/2"),
    (Model"heisenbergNNN", "S=1"),
    (Model"hubbardNNN", "Electron"),
  ]

@testset "Truncate/Compress InfiniteCanonicalMPO, H=$(model[1]), qbs=$qns, Ncell=$Ncell, NNN=$NNN" for model in models, qns in [false,true], Ncell in [1,3], NNN in [1,4]
    eps=NNN*1e-14
    initstate(n) = isodd(n) ? "↑" : "↓"
    s = infsiteinds(model[2], Ncell; initstate, conserve_qns=qns)
    Hi = InfiniteMPO(model[1](), s;NNN=NNN)

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

nothing