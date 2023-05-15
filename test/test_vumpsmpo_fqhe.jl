using ITensors
using ITensorInfiniteMPS
using Test
using Random

N = 6
model = Model"fqhe_2b_pot"()
model_params = (Vs= [1.0, 1, 0.1], Ly = 6.0, prec = 1e-8)
function initstate(n)
	return (n%3 == 0) ? 2 : 1
end
p = 1
q = 3
conserve_momentum = true

function fermion_momentum_translator(i::Index, n::Integer; N=N)
  ts = tags(i)
  translated_ts = ITensorInfiniteMPS.translatecelltags(ts, n)
  new_i = replacetags(i, ts => translated_ts)
  for j in 1:length(new_i.space)
    ch = new_i.space[j][1][1].val
    mom = new_i.space[j][1][2].val
    new_i.space[j] = Pair(QN(("Nf", ch), ("NfMom", mom + n * N * ch)), new_i.space[j][2])
  end
  return new_i
end

function ITensors.space(::SiteType"FermionK", pos::Int; p=1, q=1, conserve_momentum=true)
  if !conserve_momentum
    return [QN("Nf", -p) => 1, QN("Nf", q - p) => 1]
  else
    return [
      QN(("Nf", -p), ("NfMom", -p * pos)) => 1,
      QN(("Nf", q - p), ("NfMom", (q - p) * pos)) => 1,
    ]
  end
end

function ITensors.op!(Op::ITensor, opname::OpName, ::SiteType"FermionK", s::Index...)
  return ITensors.op!(Op, opname, SiteType("Fermion"), s...)
end


#Currently, VUMPS cannot give the right result as the subspace expansion is too small
#This is meant to test the generalized translation rules
@testset "vumpsmpo_fqhe" begin
  Random.seed!(1234)
 
  # VUMPS arguments
  cutoff = 1e-8
  maxdim = 3
  tol = 1e-8
  maxiter = 50
  outer_iters = 1

  @testset "VUMPS/TDVP with: multisite_update_alg = $multisite_update_alg, conserve_qns = $conserve_qns, nsites = $nsites" for multisite_update_alg in
                                                                                                                               [
      "sequential"
    ],
    conserve_qns in [true],
    nsites in [6],
    time_step in [-Inf]

    vumps_kwargs = (; multisite_update_alg, tol, maxiter, outputlevel=1, time_step)
    subspace_expansion_kwargs = (; cutoff, maxdim)

    # s = infsiteinds("FermionK",nsites;initstate,translator=fermion_momentum_translator,p,q,conserve_momentum)
    s = infsiteinds("FermionK", N; translator=fermion_momentum_translator, initstate, conserve_momentum, p, q);
    ψ = InfMPS(s, initstate)

    Hmpo = InfiniteMPO(model, s; model_params...)
    Hc=orthogonalize(Hmpo)
    # Alternate steps of running VUMPS and increasing the bond dimension
    ψ = tdvp(Hmpo, ψ; vumps_kwargs...)
    for _ in 1:outer_iters
      println("Subspace expansion")
      ψ = @time subspace_expansion(ψ, Hmpo; subspace_expansion_kwargs...)
      println("TDVP")
      ψ = @time tdvp(Hmpo, ψ; vumps_kwargs...)
    end
  end
end
