import ITensorMPOCompression: ac_qx, forward
#--------------------------------------------------------------------------------------------
#
#  Functions for bringing an iMPO into left or right canonical form
#

#
#  Outer routine simply established upper or lower regular forms
#
@doc """
    orthogonalize!(H::InfiniteMPO;kwargs...)

Bring `CelledVector` representation of an infinite MPO into left or right canonical form using 
block respecting QR iteration as described in section Vi B and Alogrithm 3 of:
> Daniel E. Parker, Xiangyu Cao, and Michael P. Zaletel Phys. Rev. B 102, 035147
If you intend to also call `truncate!` then do not bother calling `orthogonalize!` beforehand, as `truncate!` will do this automatically and ensure the correct handling of that gauge transforms.

# Arguments
- H::InfiniteMPO which is `CelledVector` of MPO matrices. `CelledVector` and `InfiniteMPO` are defined in the `ITensorInfiniteMPS` module.

# Keywords
- `orth::orth_type = left` : choose `left` or `right` canonical form
- `rr_cutoff::Float64 = -1.0` : cutoff for rank revealing QX which removes zero pivot rows and columns. All rows with max(abs(R[r,:]))<rr_cutoff are considered zero and removed. rr_cutoff=-11.0 indicate no rank reduction.

# Returns
- Vector{ITensor} with the gauge transforms between the input and output iMPOs

# Examples
```julia
julia> using ITensors, ITensorMPOCompression, ITensorInfiniteMPS
julia> initstate(n) = "↑";
julia> sites = infsiteinds("S=1/2", 1;initstate, conserve_szparity=false)
1-element CelledVector{Index{Int64}, typeof(translatecelltags)}:
 (dim=2|id=326|"S=1/2,Site,c=1,n=1")
#
# This makes H directly, bypassing autoMPO.  (AutoMPO is too smart for this
# demo, it makes maximally reduced MPOs right out of the box!)
#
julia> H=transIsing_MPO(sites,NNN);
julia> get_Dw(H)
1-element Vector{Int64}:
 17
julia> orthogonalize!(H;orth=right,rr_cutoff=1e-15);
julia> get_Dw(H)
1-element Vector{Int64}:
 14
julia> orthogonalize!(H;orth=left,rr_cutoff=1e-15);
julia> get_Dw(H)
 1-element Vector{Int64}:
  13
julia> isortho(H,left)
true


```

"""
function ac_orthogonalize!(H::reg_form_iMPO, lr::orth_type; verbose=false, kwargs...)
  gauge_fix!(H)

  N = length(H)
  #
  #  Init gauge transform with unit matrices.
  #
  Gs = CelledVector{ITensor}(undef, N)
  Rs = CelledVector{ITensor}(undef, N)
  for n in 1:N
    ln = lr == left ? H[n].iright : dag(H[n].iright) #get the forward link index
    Gs[n] = δ(Float64, dag(ln), ln')
  end

  if verbose
    previous_Dw = Base.max(get_Dw(H)...)
    println("niter  Dw  eta\n")
  end

  eps = 1e-13
  niter = 0
  max_iter = 40
  loop = true
  rng = sweep(H, lr)
  dn = lr == left ? 0 : -1 #index shift for Gs[n]
  while loop
    eta = 0.0
    for n in rng
      H[n], Rs[n], etan = ac_qx_step!(H[n], lr, eps; kwargs...)
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
    loop = eta > 1e-13 && niter < max_iter
  end
  return Gs
end

function ac_qx_step!(Ŵ::reg_form_Op, lr::orth_type, eps::Float64; kwargs...)
  Q̂, R, iq, p = ac_qx(Ŵ, lr; cutoff=1e-14, qprime=true, kwargs...) # r-Q-qx qx-RL-c
  #
  #  How far are we from RL==Id ?
  #
  if dim(forward(Ŵ, lr)) == dim(iq)
    eta = RmI(R, p) #Different handling for dense and block-sparse
  else
    eta = 99.0 #Rank reduction occured so keep going.
  end
  return Q̂, R, eta
end

#
#  Evaluate norm(R-I)
#
RmI(R::ITensor, perms) = RmI(tensor(R), perms)
function RmI(R::DenseTensor, perm::Vector{Int64})
  Rmp = matrix(R)[:, perm]
  return norm(Rmp - Matrix(LinearAlgebra.I, size(Rmp)))
end

function RmI(R::BlockSparseTensor, perms::Vector{Vector{Int64}})
  @assert nnzblocks(R) == length(perms)
  eta2 = 0.0
  for (n, b) in enumerate(nzblocks(R))
    bv = ITensors.blockview(R, b)
    Rp = bv[:, perms[n]] #un-permute the columns
    eta2 += norm(Rp - Matrix(LinearAlgebra.I, size(Rp)))^2
  end
  return sqrt(eta2)
end
