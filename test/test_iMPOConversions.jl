using ITensors
using ITensorInfiniteMPS
using Test
#
# InfiniteMPO has dangling links at the end of the chain.  We contract these on the outside
#   with l,r terminating vectors, to make a finite lattice MPO.
#
function terminate(h::InfiniteMPO)::MPO
  Ncell = nsites(h)
  # left termination vector
  il0 = commonind(h[1], h[0])
  l = ITensor(0.0, il0)
  l[il0 => dim(il0)] = 1.0 #assuming lower reg form in h
  # right termination vector
  iln = commonind(h[Ncell], h[Ncell + 1])
  r = ITensor(0.0, iln)
  r[iln => 1] = 1.0 #assuming lower reg form in h
  # build up a finite MPO
  hf = MPO(Ncell)
  hf[1] = dag(l) * h[1] #left terminate
  hf[Ncell] = h[Ncell] * dag(r) #right terminate
  for n in 2:(Ncell - 1)
    hf[n] = h[n] #fill in the bulk.
  end
  return hf
end
#
# Terminate and then call expect
# for inf ψ and finite h, which is already supported in src/infinitecanonicalmps.jl
#
function ITensors.expect(ψ::InfiniteCanonicalMPS, h::InfiniteMPO)
  return expect(ψ, terminate(h)) #defer to src/infinitecanonicalmps.jl
end


@testset verbose = true "InfiniteMPOMatrix -> InfiniteMPO" begin
  ferro(n) = "↑"
  antiferro(n) = isodd(n) ? "↑" : "↓"

  models = [(Model"heisenbergNNN"(), "S=1/2"), (Model"hubbardNNN"(), "Electron")]
  @testset "H=$model, Ncell=$Ncell, NNN=$NNN, Antiferro=$Af, qns=$qns" for (model, site) in
                                                                           models,
    qns in [false, true],
    Ncell in 2:6,
    NNN in 1:(Ncell - 1),
    Af in [true, false]

    if isodd(Ncell) && Af #skip test since Af state does fit inside odd cells.
      continue
    end
    initstate(n) = Af ? antiferro(n) : ferro(n)
    model_kwargs = (NNN=NNN,)
    s = infsiteinds(site, Ncell; initstate, conserve_qns=qns)
    ψ = InfMPS(s, initstate)
    Hi = InfiniteMPO(model, s; model_kwargs...)
    Hs = InfiniteSum{MPO}(model, s; model_kwargs...)
    Es = expect(ψ, Hs)
    Ei = expect(ψ, Hi)
    #@show Es Ei
    @test sum(Es[1:(Ncell - NNN)]) ≈ Ei atol = 1e-14
  end

  @testset "FQHE Hamitonian" begin
    include("hamiltonians/fqhe.jl")
    ψ = InfMPS(s, initstate)
    Hs = InfiniteSum{MPO}(fqhe_model, s; fqhe_model_params...)
    Hi = InfiniteMPO(fqhe_model, s, fermion_momentum_translator; fqhe_model_params...)
    Es = expect(ψ, Hs)
    Ei = expect(ψ, Hi)
    # @show Es Ei
    @test Es[1] ≈ Ei atol = 1e-14
  end
end
nothing
