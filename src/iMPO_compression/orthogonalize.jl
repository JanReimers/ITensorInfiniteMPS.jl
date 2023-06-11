import Base: inv

#--------------------------------------------------------------------------------------------
#
#  Functions for bringing an iMPO into left or right canonical form
#

#
#  Outer routine simply established upper or lower regular forms
#
function ITensors.orthogonalize(Hi::reg_form_iMPO;kwargs...)
  HL=copy(Hi) #not HL yet, but will be after two ortho calls.
  G0L=orthogonalize!(HL, left; kwargs...)
  HR = copy(HL)
  GLR = orthogonalize!(HR,right; kwargs...)
  G0LR=full_ortho_gauge(G0L,GLR)
  #
  #  At this point HL is likely to have a larger bond dimension than HR because the second sweep matters for reducing Dw.
  #  Truncation will make this discrepency moot.
  #  If we need a reduced HL for some reason then one more sweep as below can be uncommented.
  # HL=copy(HR)
  # GRL = orthogonalize!(HL,left; kwargs...)
  return HL,HR,GLR,G0LR
end


function orthogonalize!(H::reg_form_iMPO, lr::orth_type; eps_qr=1e-13, max_iter=40, verbose=false, kwargs...)
  gauge_fix!(H)

  N = length(H)
  #
  #  Init gauge transform with unit matrices.
  #
  Gs = CelledVector{ITensor}(undef, N, translator(H))
  for n in 1:N
    ln = lr == left ? H[n].iright : dag(H[n].iright) #get the forward link index
    Gs[n] = δ(Float64, dag(ln), ln')
  end

  if verbose
    println("niter  Dw  eta\n")
  end

  niter = 0
  rng = sweep(H, lr)
  dn = lr == left ? 0 : -1 #index shift for Gs[n]
  Rs = CelledVector{ITensor}(undef, N, translator(H))
  while true
    eta = 0.0
    for n in rng
      H[n], Rs[n], etan = ac_qx_step!(H[n], lr; kwargs...)
      Gs[n + dn] = noprime(Rs[n] * Gs[n + dn])  #  Update the accumulated gauge transform
      @mpoc_assert order(Gs[n + dn]) == 2 #This will fail if the indices somehow got messed up.
      eta = Base.max(eta, etan)
    end
    #
    #  H now contains all the Qs.  We need to transfer the R's to the neighbouring sites.
    #
    for n in rng
      H[n] *= noprime(Rs[n - rng.step])
      check(H[n])
    end
    niter += 1
    if verbose
      println("$niter $(Base.max(get_Dw(H)...)) $eta")
    end
    (eta > eps_qr && niter < max_iter) || break
  end
  return Gs
end

function ac_qx_step!(Ŵ::reg_form_Op, lr::orth_type; kwargs...)
  Q̂, R, iq, Rp = ac_qx(Ŵ, lr; cutoff=1e-14, qprime=true, return_Rp=true, kwargs...) # r-Q-qx qx-RL-c
  #
  #  How far are we from RL==Id ?
  #
  if dim(forward(Ŵ, lr)) == dim(iq)
    eta = RmI(Rp) #Different handling for dense and block-sparse
  else
    eta = 99.0 #Rank reduction occured so keep going.
  end
  return Q̂, R, eta
end

#
#  Evaluate norm(R-I)
#
RmI(R::ITensor) = RmI(tensor(R))

function RmI(R::DenseTensor)
  return norm(R - Matrix(LinearAlgebra.I, size(R)))
end

function RmI(R::BlockSparseTensor)
  eta2 = 0.0
  for b in nzblocks(R)
    eta2 += RmI(blockview(R, b))^2
  end
  return sqrt(eta2)
end

#
#  Assumes we first did right orth H0-->HR and the a left orth HR-->HL
#  The H's should satisfy: H0[K]*G[K]-G[k-1]*HL[k]
#
function full_ortho_gauge(G0L::CelledVector{ITensor},GLR::CelledVector{ITensor})
  @assert length(G0L)==length(GLR)
  @assert translator(G0L)==translator(GLR)
  N=length(G0L)
  G0R=CelledVector{ITensor}(undef,N,translator(G0L))
  for k in 1:N
      G0R[k]=Base.inv(G0L[k])*GLR[k]
      @assert order(G0R[k])==2
  end
  return G0R
end

#
#  Penrose inversin code used for getting the H0 --> HR gauge transform
#
function Base.inv(A::ITensor;tol=1e-12,kwargs...)::ITensor
@assert order(A)==2

U,s,V=svd(A,ind(A,1);kwargs...)
if minimum(diag(s))<tol
  @warn("Trying to solve near singular system. diag(s)=$(diag(s))")
end
return dag(V)*invdiag(s)*dag(U)
end

function invdiag(s::ITensor)
  return itensor(invdiag(tensor(s)))
end

function invdiag(s::DiagTensor)
  # creating a DiagTensor directly seems to be very diffficult
  sinv=tensor(diagITensor(dag(inds(s))))
  for i in 1:diaglength(s)
      s1=1.0/NDTensors.getdiagindex(s,i)
      NDTensors.setdiagindex!(sinv,s1,i)
  end
  return sinv
end

function invdiag(s::DiagBlockSparseTensor)
  sinv=DiagBlockSparseTensor(nzblocks(s),dag(inds(s)))
  for i in 1:diaglength(s)
      s1=1.0/NDTensors.getdiagindex(s,i)
      NDTensors.setdiagindex!(sinv,s1,i)
  end
  return sinv
end


