using ITensors
using ITensorInfiniteMPS
using Test
using Random

function expect_three_site(ψ::MPS, h::ITensor, n::Int)
  ϕ = ψ[n] * ψ[n + 1] * ψ[n + 2]
  return inner(ϕ, (apply(h, ϕ)))
end

verbose=false

#Ref time is 21.6s with negligible compilation time
# @testset verbose=true "vumpsmpo_ising, H=$H_type" for  H_type in [InfiniteMPOMatrix, InfiniteMPO]
#   Random.seed!(1234)

#   model = Model("ising")
#   model_kwargs = (J=1.0, h=1.2)

#   # VUMPS arguments
#   cutoff = 1e-8
#   maxdim = 20
#   tol = 1e-8
#   maxiter = 50
#   outer_iters = 3

#   initstate(n) = "↑"

#   # DMRG arguments
#   Nfinite = 100
#   sfinite = siteinds("S=1/2", Nfinite; conserve_szparity=true)
#   Hfinite = MPO(model, sfinite; model_kwargs...)
#   ψfinite = randomMPS(sfinite, initstate)

#   sweeps = Sweeps(20)
#   setmaxdim!(sweeps, 10)
#   setcutoff!(sweeps, 1E-10)
#   energy_finite_total, ψfinite = dmrg(Hfinite, ψfinite, sweeps; outputlevel=0)
#   Szs_finite = expect(ψfinite, "Sz")

#   nfinite = Nfinite ÷ 2
#   hnfinite = ITensor(model, sfinite[nfinite], sfinite[nfinite + 1]; model_kwargs...)
#   orthogonalize!(ψfinite, nfinite)
#   energy_finite = expect_three_site(ψfinite, hnfinite, nfinite)

#   @testset "VUMPS/TDVP with: multisite_update_alg = $multisite_update_alg, conserve_qns = $conserve_qns, nsites = $nsites" for multisite_update_alg in
#                                                                                                                                [
#       "sequential", "parallel"
#     ],
#     conserve_qns in [true, false],
#     nsites in [1, 2],
#     time_step in [-Inf]
   

#     vumps_kwargs = (
#       multisite_update_alg=multisite_update_alg,
#       tol=tol,
#       maxiter=maxiter,
#       outputlevel=0,
#       time_step=time_step,
#     )
#     subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)

#     s = infsiteinds("S=1/2", nsites; initstate, conserve_szparity=conserve_qns)
#     ψ = InfMPS(s, initstate)

#     Hmpo = H_type(model, s; model_kwargs...)
#     # Alternate steps of running VUMPS and increasing the bond dimension
#     ψ = tdvp(Hmpo, ψ; vumps_kwargs...)
#     for _ in 1:outer_iters
#       if verbose
#         println("Subspace expansion")
#         ψ = @time subspace_expansion(ψ, Hmpo; subspace_expansion_kwargs...)
#         println("TDVP")
#         ψ = @time tdvp(Hmpo, ψ; vumps_kwargs...)
#       else
#         ψ = subspace_expansion(ψ, Hmpo; subspace_expansion_kwargs...)
#         ψ = tdvp(Hmpo, ψ; vumps_kwargs...)      
#       end
#     end

#     @test norm(
#       contract(ψ.AL[1:nsites]..., ψ.C[nsites]) - contract(ψ.C[0], ψ.AR[1:nsites]...)
#     ) ≈ 0 atol = 1e-5
#     #@test contract(ψ.AL[1:nsites]..., ψ.C[nsites]) ≈ contract(ψ.C[0], ψ.AR[1:nsites]...)

#     H = InfiniteSum{MPO}(model, s; model_kwargs...)
#     energy_infinite = expect(ψ, H)
#     Szs_infinite = [expect(ψ, "Sz", n) for n in 1:nsites]

#     @test energy_finite ≈ sum(energy_infinite) / nsites rtol = 1e-4
#     @test Szs_finite[nfinite:(nfinite + nsites - 1)] ≈ Szs_infinite rtol = 1e-3
#   end
# end

# @testset "vumpsmpo_extendedising, H=$H_type" for H_type in [InfiniteMPOMatrix,InfiniteMPO]
#   Random.seed!(1234)

#   model = Model"ising_extended"()
#   model_kwargs = (J=1.0, h=1.1, J₂=0.2)

#   # VUMPS arguments
#   cutoff = 1e-8
#   maxdim = 20
#   tol = 1e-8
#   maxiter = 20
#   outer_iters = 4

#   initstate(n) = "↑"

#   # DMRG arguments
#   Nfinite = 100
#   sfinite = siteinds("S=1/2", Nfinite; conserve_szparity=true)
#   Hfinite = MPO(model, sfinite; model_kwargs...)
#   ψfinite = randomMPS(sfinite, initstate)
#   nsweeps = 20
#   energy_finite_total, ψfinite = dmrg(
#     Hfinite, ψfinite; nsweeps, maxdims=10, cutoff=1e-10, outputlevel=0
#   )
#   Szs_finite = expect(ψfinite, "Sz")

#   nfinite = Nfinite ÷ 2
#   hnfinite = ITensor(
#     model, sfinite[nfinite], sfinite[nfinite + 1], sfinite[nfinite + 2]; model_kwargs...
#   )
#   orthogonalize!(ψfinite, nfinite)
#   energy_finite = expect_three_site(ψfinite, hnfinite, nfinite)

#   for multisite_update_alg in ["sequential"],
#     conserve_qns in [true, false],
#     nsites in [1, 2],
#     time_step in [-Inf]

#     vumps_kwargs = (
#       multisite_update_alg=multisite_update_alg,
#       tol=tol,
#       maxiter=maxiter,
#       outputlevel=0,
#       time_step=time_step,
#     )
#     subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)

#     s = infsiteinds("S=1/2", nsites; conserve_szparity=conserve_qns, initstate)
#     ψ = InfMPS(s, initstate)

#     Hmpo = H_type(model, s; model_kwargs...)
#     # Alternate steps of running VUMPS and increasing the bond dimension
#     ψ = tdvp(Hmpo, ψ; vumps_kwargs...)
#     for _ in 1:outer_iters
#       ψ = subspace_expansion(ψ, Hmpo; subspace_expansion_kwargs...)
#       ψ = tdvp(Hmpo, ψ; vumps_kwargs...)
#     end

#     @test norm(
#       contract(ψ.AL[1:nsites]..., ψ.C[nsites]) - contract(ψ.C[0], ψ.AR[1:nsites]...)
#     ) ≈ 0 atol = 1e-5

#     H = InfiniteSum{MPO}(model, s; model_kwargs...)
#     energy_infinite = expect(ψ, H)
#     Szs_infinite = [expect(ψ, "Sz", n) for n in 1:nsites]

#     @test energy_finite ≈ sum(energy_infinite) / nsites rtol = 1e-4
#     @test Szs_finite[nfinite:(nfinite + nsites - 1)] ≈ Szs_infinite rtol = 1e-3
#   end
# end

# @testset "vumpsmpo_extendedising_translator, H=$H_type" for H_type in [InfiniteMPOMatrix,InfiniteMPO]
#   Random.seed!(1234)

#   model = Model("ising_extended")
#   model_kwargs = (J=1.0, h=1.1, J₂=0.2)

#   # VUMPS arguments
#   cutoff = 1e-8
#   maxdim = 20
#   tol = 1e-8
#   maxiter = 20
#   outer_iters = 4

#   initstate(n) = "↑"

#   # DMRG arguments
#   Nfinite = 100
#   sfinite = siteinds("S=1/2", Nfinite; conserve_szparity=true)
#   Hfinite = MPO(model, sfinite; model_kwargs...)
#   ψfinite = randomMPS(sfinite, initstate)
#   sweeps = Sweeps(20)
#   setmaxdim!(sweeps, 10)
#   setcutoff!(sweeps, 1E-10)
#   energy_finite_total, ψfinite = dmrg(Hfinite, ψfinite, sweeps; outputlevel=0)
#   Szs_finite = expect(ψfinite, "Sz")

#   nfinite = Nfinite ÷ 2
#   hnfinite = ITensor(
#     model, sfinite[nfinite], sfinite[nfinite + 1], sfinite[nfinite + 2]; model_kwargs...
#   )
#   orthogonalize!(ψfinite, nfinite)
#   energy_finite = expect_three_site(ψfinite, hnfinite, nfinite)

#   temp_translatecell(i::Index, n::Integer) = ITensorInfiniteMPS.translatecelltags(i, n)

#   for multisite_update_alg in ["sequential"],
#     conserve_qns in [true, false],
#     nsite in [1, 2, 3],
#     time_step in [-Inf]

#     if nsite > 1 && isodd(nsite) && conserve_qns
#       # Parity conservation not commensurate with odd number of sites.
#       continue
#     end

#     vumps_kwargs = (
#       multisite_update_alg=multisite_update_alg,
#       tol=tol,
#       maxiter=maxiter,
#       outputlevel=0,
#       time_step=time_step,
#     )
#     subspace_expansion_kwargs = (cutoff=cutoff, maxdim=maxdim)

#     s_bis = infsiteinds("S=1/2", nsite; conserve_szparity=conserve_qns, initstate)
#     s = infsiteinds(
#       "S=1/2",
#       nsite;
#       conserve_szparity=conserve_qns,
#       initstate,
#       translator=temp_translatecell,
#     )
#     ψ = InfMPS(s, initstate)

#     Hmpo = InfiniteMPOMatrix(model, s; model_kwargs...)
#     # Alternate steps of running VUMPS and increasing the bond dimension
#     ψ = tdvp(Hmpo, ψ; vumps_kwargs...)
#     for _ in 1:outer_iters
#       ψ = subspace_expansion(ψ, Hmpo; subspace_expansion_kwargs...)
#       ψ = tdvp(Hmpo, ψ; vumps_kwargs...)
#     end

#     @test norm(
#       contract(ψ.AL[1:nsite]..., ψ.C[nsite]) - contract(ψ.C[0], ψ.AR[1:nsite]...)
#     ) ≈ 0 atol = 1e-5

#     H = InfiniteSum{MPO}(model, s; model_kwargs...)
#     energy_infinite = expect(ψ, H)
#     Szs_infinite = [expect(ψ, "Sz", n) for n in 1:nsite]

#     @test energy_finite ≈ sum(energy_infinite) / nsite rtol = 1e-4
#     @test Szs_finite[nfinite:(nfinite + nsite - 1)] ≈ Szs_infinite rtol = 1e-3

#     #@test tags(s[nsite + 1]) == tags(s_bis[1 + 2nsite])
#     @test ITensorInfiniteMPS.translator(ψ) == temp_translatecell
#     @test ITensorInfiniteMPS.translator(s) == temp_translatecell
#     @test ITensorInfiniteMPS.translator(Hmpo) == temp_translatecell
#   end
# end

#
# Testing for non-rectangular MPOs using Heisenberg model with different number of neighbouring interactions
# for each site in the unit cell.  This is supposed to trip up any code that assumes iMPO site ops (Ws) are always square.
# For now we go step by testing 1) raw iMPOs, 2) gauge fixed only, 3) orthogonalized (which requires gauge fixing), 
# 4) truncated (whihc requires orthogonalizing)
# These take a long time to run.  In future we should only do subspace expand to D=4 to speed things up.  
# THis will change all the expected energies.
#
import ITensorInfiniteMPS: reg_form_iMPO, gauge_fix!,get_Dw

#  
#     NNN[i] = Number of Nearest Neightbours for site i in the unit cell
#
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

function ITensorInfiniteMPS.unit_cell_terms(::Model"hubbardNNNs"; NNNs::Vector{Int64},U=0.25,t=1.0,V=0.5)
  opsum = OpSum()
  for i in eachindex(NNNs)
    opsum += (U, "Nupdn", i)
    for n in 1:NNNs[i]
      tn, Vn = t / n, V / n
      opsum += -tn, "Cdagup", i, "Cup", i + n
      opsum += -tn, "Cdagup", i + n, "Cup", i
      opsum += -tn, "Cdagdn", i, "Cdn", i + n
      opsum += -tn, "Cdagdn", i + n, "Cdn", i
      opsum += Vn, "Ntot", i, "Ntot", i + n
    end
  end
  return opsum
end

function mean_var(e::Vector{Float64})
  n=length(e)
  μ=sum(e)/n
  σ=n>1 ? sqrt(sum((e .- μ) .^ 2) / (n - 1)) : 0.0
  return μ,σ
end

# exact e=-0.443147181 
tests=[
  # Higher precision versions.
  # [2,[1,1],5,-0.4431377],
  # [4,[1,1],5,-0.4431377],
  # [4,[1,1,1,1],5,-0.4431377],
  # [2,[1,2],5,-0.4005882],
  # [2,[2,1],5,-0.4005882],
  # [4,[1,2],5,-0.4005882],
  # [4,[1,2,1,2],5,-0.4005882],
  # [4,[2,1,2,1],5,-0.4005882],
  # [4,[1,2,3,4],4,-0.4081936],
  # [4,[2,3,4,1],4,-0.4081936],
  # [4,[3,4,1,2],4,-0.4081936],
  # [4,[4,1,2,3],4,-0.4081936],

  # medium precision, short run time.
  # [2,[1,1],4,-0.443073],
  # [4,[1,1],4,-0.443073],
  # [4,[1,1,1,1],4,-0.443073],
  # [2,[1,2],4,-0.4005530],
  # [2,[2,1],4,-0.4005530],
  # [4,[1,2],4,-0.4005530],
  # [4,[1,2,1,2],4,-0.4005530],
  # [4,[2,1,2,1],4,-0.4005530],
  # [4,[1,2,3,4],3,-0.4080177], #These last 4 should all be equal, Check MPO bond spectra
  # [4,[2,3,4,1],3,-0.4081479],
  # [4,[3,4,1,2],3,-0.4077748],
  # [4,[4,1,2,3],3,-0.4076770],

  # Low precision, shortest run time.
  # [2,[1,1],2,Model"heisenbergNNNs"(),"S=1/2",-0.4410585],
  # [4,[1,1],2,Model"heisenbergNNNs"(),"S=1/2",-0.4410585],
  # [4,[1,1,1,1],2,Model"heisenbergNNNs"(),"S=1/2",-0.4410585],
  # [2,[1,2],2,Model"heisenbergNNNs"(),"S=1/2",-0.3990024],
  # [2,[2,1],2,Model"heisenbergNNNs"(),"S=1/2",-0.3990024],
  # [4,[1,2],2,Model"heisenbergNNNs"(),"S=1/2",-0.3990024],
  # [4,[2,1],2,Model"heisenbergNNNs"(),"S=1/2",-0.3990024],
  # [4,[1,2,1,2],2,Model"heisenbergNNNs"(),"S=1/2",-0.3990024],
  # [4,[2,1,2,1],2,Model"heisenbergNNNs"(),"S=1/2",-0.3990024],
  # [4,[1,2,3,4],1,Model"heisenbergNNNs"(),"S=1/2",-0.3835463], #These next 4 should all be equal, Check MPO bond spectra: DOne, spectra are identical.
  # [4,[2,3,4,1],1,Model"heisenbergNNNs"(),"S=1/2",-0.3886438],
  # [4,[3,4,1,2],1,Model"heisenbergNNNs"(),"S=1/2",-0.3833354],
  # [4,[4,1,2,3],1,Model"heisenbergNNNs"(),"S=1/2",-0.3833374],
  # [4,[4,1,1,1],2,Model"heisenbergNNNs"(),"S=1/2",-0.43000425],  #again these next should all be the same; Bond spectra verified identical.
  # [4,[1,4,1,1],2,Model"heisenbergNNNs"(),"S=1/2",-0.43002159], 
  # [4,[1,1,4,1],2,Model"heisenbergNNNs"(),"S=1/2",-0.43023817], 
  # [4,[1,1,1,4],2,Model"heisenbergNNNs"(),"S=1/2",-0.43000425], 
  # [2,[4,1],2,-0.42260203], 
  # [2,[1,4],2,-0.422503386], 
  # [4,[4,1],2,-0.4223896], 
  # [4,[1,4],2,-0.42236642], 
  # [4,[1,4,1,4],2,-0.42236642], 
  # [4,[4,1,4,1],2,-0.4223896], 

  [2,[1,1],2,Model"hubbardNNNs"(),"Electron",-0.8126998108567591], #easy commensurate
  #
  #  See energies converge as N increases.  Presumably towards an incommmensurate GS.
  #
  [2,[3,1]    ,2,Model"hubbardNNNs"(),"Electron",-0.7756839346113128],
  [4,[3,1]    ,2,Model"hubbardNNNs"(),"Electron",-0.7734956745259833],
  [4,[3,1,3,1],2,Model"hubbardNNNs"(),"Electron",-0.7734956745259833], #Double cell for H should change nothing
  [6,[3,1]    ,2,Model"hubbardNNNs"(),"Electron",-0.7681634027405047],
  [8,[3,1]    ,2,Model"hubbardNNNs"(),"Electron",-0.7675595918275887],
  [2,[1,3]    ,2,Model"hubbardNNNs"(),"Electron",-0.7660642672670517],
  [4,[1,3]    ,2,Model"hubbardNNNs"(),"Electron",-0.7659048521284240],
  [4,[1,3,1,3],2,Model"hubbardNNNs"(),"Electron",-0.7659048521284240], #Double cell for H should change nothing
  [6,[1,3]    ,2,Model"hubbardNNNs"(),"Electron",-0.7658535005422838],
  [8,[1,3]    ,2,Model"hubbardNNNs"(),"Electron",-0.7658278899976771],
  #
  #  Try and trigger the subspcae expansion error with long range interactions.
  #
  [2,[7,1],2,Model"hubbardNNNs"(),"Electron",-0.6976465308516251],
  [2,[1,7],2,Model"hubbardNNNs"(),"Electron",-0.6783064305516868],
 

  ]

@testset verbose=true "vumps for truncated rectangular iMPOs, N=$(test[1]), NNNs=$(test[2]), MPO rep.=$H_type, qns=$qns, alg=$alg" for test in tests, H_type in [InfiniteMPO], qns in [true], alg=["parallel"]
  initstate(n) = isodd(n) ? "↑" : "↓"
  N,NNNs,n_expansions,model,stype,e_expected=test[1],test[2],test[3],test[4],test[5],test[6]
  tol=1e-5
  vumps_kwargs = (
      multisite_update_alg=alg,
      tol=tol,
      maxiter=50,
      outputlevel=0,
      return_e=true,
      time_step=-Inf,
    )
  sites = infsiteinds(stype, N; initstate, conserve_qns=qns)
  ψ = InfMPS(sites, initstate)
  H = H_type(model, sites; NNNs=NNNs)
  Ht,ss=truncate(H)
  @show get_Dw(Ht.AR)
  ψ,(eᴸ, eᴿ) = tdvp(Ht, ψ; vumps_kwargs...)
  for _ in 1:n_expansions
      ψ = subspace_expansion(ψ, Ht; cutoff=1e-8,maxdim=32)
      ψ,(eᴸ, eᴿ) = tdvp(Ht, ψ; vumps_kwargs...)      
  end
  eps=2e-6*N
  μᴸ,σᴸ=mean_var(eᴸ)
  μᴿ,σᴿ=mean_var(eᴿ)
  μ,σ=mean_var(vcat(eᴸ,eᴿ))
  # @show eᴸ eᴿ μᴸ,σᴸ μᴿ,σᴿ μ,σ
  # @show μᴸ-e_expected μᴿ-e_expected μ-e_expected
  @show μ,μ-e_expected
  @test σᴸ < eps
  @test σᴿ < eps
  @test σ < eps
  @test μᴸ ≈ e_expected atol = eps
  @test μᴿ ≈ e_expected atol = eps
  @test μ ≈ e_expected atol = eps
  
end

# @testset verbose=false "vumps for rectangular iMPOs, N=$(test[1]), NNNs=$(test[2]), MPO rep.=$H_type, qns=$qns, alg=$alg" for test in tests, H_type in [InfiniteMPO], qns in [true], alg=["parallel"]
#   initstate(n) = isodd(n) ? "↑" : "↓"
#   N,NNNs,n_expansions,e_expected=test[1],test[2],test[3],test[4]
#   # qns=true
#   tol=1e-5
#   vumps_kwargs = (
#       multisite_update_alg=alg,
#       tol=tol,
#       maxiter=50,
#       outputlevel=0,
#       return_e=true,
#       time_step=-Inf,
#     )
#   sites = infsiteinds("S=1/2", N; initstate, conserve_qns=qns)
#   ψ = InfMPS(sites, initstate)
#   Hmpo = H_type(Model"heisenbergNNNs"(), sites; NNNs=NNNs)
#   ψ,(eᴸ, eᴿ) = tdvp(Hmpo, ψ; vumps_kwargs...)
#   for _ in 1:n_expansions
#       ψ = subspace_expansion(ψ, Hmpo; cutoff=1e-8,maxdim=32)
#       ψ,(eᴸ, eᴿ) = tdvp(Hmpo, ψ; vumps_kwargs...)      
#   end
#   eps=2e-6*N
#   μᴸ,σᴸ=mean_var(eᴸ)
#   μᴿ,σᴿ=mean_var(eᴿ)
#   μ,σ=mean_var(vcat(eᴸ,eᴿ))
#   # @show eᴸ eᴿ μᴸ,σᴸ μᴿ,σᴿ μ,σ
#   # @show μᴸ-e_expected μᴿ-e_expected μ-e_expected
#   @show μ,μ-e_expected
#   @test σᴸ < eps
#   @test σᴿ < eps
#   @test σ < eps
#   @test μᴸ ≈ e_expected atol = eps
#   @test μᴿ ≈ e_expected atol = eps
#   @test μ ≈ e_expected atol = eps
  
# end


# @testset verbose=true "vumps for gauge fixed rectangular iMPOs, N=$(test[1]), NNNs=$(test[2]), MPO rep.=$H_type, qns=$qns, alg=$alg" for test in tests, H_type in [InfiniteMPO], qns in [true], alg=["sequential","parallel"]
#   initstate(n) = isodd(n) ? "↑" : "↓"
#   N,NNNs,n_expansions,e_expected=test[1],test[2],test[3],test[4]
#   # qns=true
#   tol=1e-5
#   vumps_kwargs = (
#       multisite_update_alg=alg,
#       tol=tol,
#       maxiter=50,
#       outputlevel=0,
#       return_e=true,
#       time_step=-Inf,
#     )
#   sites = infsiteinds("S=1/2", N; initstate, conserve_qns=qns)
#   ψ = InfMPS(sites, initstate)
#   H = H_type(Model"heisenbergNNNs"(), sites; NNNs=NNNs)
#   Hrf=reg_form_iMPO(H)
#   gauge_fix!(Hrf)
#   Hmpo=InfiniteMPO(Hrf)
#   ψ,(eᴸ, eᴿ) = tdvp(Hmpo, ψ; vumps_kwargs...)
#   for _ in 1:n_expansions
#       ψ = subspace_expansion(ψ, Hmpo; cutoff=1e-8,maxdim=32)
#       ψ,(eᴸ, eᴿ) = tdvp(Hmpo, ψ; vumps_kwargs...)      
#   end
#   eps=2e-6*N
#   μᴸ,σᴸ=mean_var(eᴸ)
#   μᴿ,σᴿ=mean_var(eᴿ)
#   μ,σ=mean_var(vcat(eᴸ,eᴿ))
#   # @show eᴸ eᴿ μᴸ,σᴸ μᴿ,σᴿ μ,σ
#   # @show μᴸ-e_expected μᴿ-e_expected μ-e_expected
#   @show μ,μ-e_expected
#   @test σᴸ < eps
#   @test σᴿ < eps
#   @test σ < eps
#   @test μᴸ ≈ e_expected atol = eps
#   @test μᴿ ≈ e_expected atol = eps
#   @test μ ≈ e_expected atol = eps
  
# end

# @testset verbose=true "vumps for orthongonalized rectangular iMPOs, N=$(test[1]), NNNs=$(test[2]), MPO rep.=$H_type, qns=$qns, alg=$alg" for test in tests, H_type in [InfiniteMPO], qns in [true], alg=["sequential","parallel"]
#   initstate(n) = isodd(n) ? "↑" : "↓"
#   N,NNNs,n_expansions,e_expected=test[1],test[2],test[3],test[4]
#   tol=1e-5
#   vumps_kwargs = (
#       multisite_update_alg=alg,
#       tol=tol,
#       maxiter=50,
#       outputlevel=0,
#       return_e=true,
#       time_step=-Inf,
#     )
#   sites = infsiteinds("S=1/2", N; initstate, conserve_qns=qns)
#   ψ = InfMPS(sites, initstate)
#   H = H_type(Model"heisenbergNNNs"(), sites; NNNs=NNNs)
#   Ho=orthogonalize(H)
#   @show get_Dw(Ho.AR)
#   ψ,(eᴸ, eᴿ) = tdvp(Ho, ψ; vumps_kwargs...)
#   for _ in 1:n_expansions
#       ψ = subspace_expansion(ψ, Ho; cutoff=1e-8,maxdim=32)
#       ψ,(eᴸ, eᴿ) = tdvp(Ho, ψ; vumps_kwargs...)      
#   end
#   eps=2e-6*N
#   μᴸ,σᴸ=mean_var(eᴸ)
#   μᴿ,σᴿ=mean_var(eᴿ)
#   μ,σ=mean_var(vcat(eᴸ,eᴿ))
#   # @show eᴸ eᴿ μᴸ,σᴸ μᴿ,σᴿ μ,σ
#   # @show μᴸ-e_expected μᴿ-e_expected μ-e_expected
#   @show μ,μ-e_expected
#   @test σᴸ < eps
#   @test σᴿ < eps
#   @test σ < eps
#   @test μᴸ ≈ e_expected atol = eps
#   @test μᴿ ≈ e_expected atol = eps
#   @test μ ≈ e_expected atol = eps
  
# end


nothing