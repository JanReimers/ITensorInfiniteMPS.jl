
@doc """
    truncate!(H::InfiniteMPO;kwargs...)

Truncate a `CelledVector` representation of an infinite MPO as described in section VII and Alogrithm 5 of:
> Daniel E. Parker, Xiangyu Cao, and Michael P. Zaletel Phys. Rev. B 102, 035147
It is not nessecary (or recommended) to call the `orthogonalize!` function prior to calling `truncate!`. The `truncate!` function will do this automatically.  This is because the truncation process requires the gauge transform tensors resulting from left orthogonalizing an already right orthogonalized iMPO (or converse).  So it is better to do this internally in order to be sure the correct gauge transforms are used.

# Arguments
- H::InfiniteMPO which is a `CelledVector` of MPO matrices. `CelledVector` and `InfiniteMPO` are defined in the `ITensorInfiniteMPS` module.

# Keywords
- `orth::orth_type = left` : choose `left` or `right` canonical form for the output
- `rr_cutoff::Float64 = -1.0` : cutoff for rank revealing QX which removes zero pivot rows and columns. 
   All rows with max(abs(R[r,:]))<rr_cutoff are considered zero and removed. rr_cutoff=1.0 indicate no rank reduction.
- `cutoff::Float64 = 0.0` : Using a `cutoff` allows the SVD algorithm to truncate as many states as possible while still ensuring a certain accuracy. 
- `maxdim::Int64` : If the number of singular values exceeds `maxdim`, only the largest `maxdim` will be retained.
- `mindim::Int64` : At least `mindim` singular values will be retained, even if some fall below the cutoff
   
# Returns
- Vector{ITensor} with the diagonal gauge transforms between the input and output iMPOs
- a `bond_spectrums` object which is a `Vector{Spectrum}`

# Example
```
julia> using ITensors, ITensorMPOCompression, ITensorInfiniteMPS
julia> initstate(n) = "↑";
julia> sites = infsiteinds("S=1/2", 1;initstate, conserve_szparity=false)
1-element CelledVector{Index{Int64}, typeof(translatecelltags)}:
 (dim=2|id=224|"S=1/2,Site,c=1,n=1")
julia> H=transIsing_iMPO(sites,7);
julia> get_Dw(H)[1]
30
julia> Ss,spectrum=truncate!(H;rr_cutoff=1e-15,cutoff=1e-15);
julia> get_Dw(H)[1]
9
julia> pprint(H[1])
I 0 0 0 0 0 0 0 0 
S S S S S S S S 0 
S S S S S S S S 0 
S S S S S S S S 0 
S S S S S S S S 0 
S S S S S S S S 0 
S S S S S S S S 0 
S S S S S S S S 0 
0 S S S S S S S I 
julia> @show spectrum
spectrum = 
site  Ns   max(s)     min(s)    Entropy  Tr. Error
   1    7  0.39565   1.26e-02   0.32644  1.23e-16

```
"""
function truncate!(
  Hi::reg_form_iMPO; rr_cutoff=1e-14, kwargs...
)::Tuple{reg_form_iMPO,reg_form_iMPO,Any,Any,bond_spectrums}
  #
  #  Orthogonalize and get the gauge transforms between HL and HR
  #
  # HR=copy(H)
  # orthogonalize!(HR, left; cutoff=rr_cutoff, kwargs...)
  # HL = copy(HR)
  # Gs = orthogonalize!(HR, right; cutoff=rr_cutoff, kwargs...)
  HL,GLR,HR,G = orthogonalize(Hi;cutoff=rr_cutoff,kwargs...)
  #
  #  Now compress and gauge transforms and apply the similarity transform to HL and HR.
  #
  N = length(Hi)
  ss = bond_spectrums(undef, N)
  Ss = CelledVector{ITensor}(undef, N,translator(Hi))
  for k in 1:N
    #prime the right index of G so that indices can be distinguished when N==1.
    GLR[k] = prime(GLR[k],HR[k+1].ileft)
    U, Ss[k], V, ss[k] = truncateG(GLR[k], dag(HL[k].iright) ; kwargs...)
    HL[k] *= U
    HL[k + 1] *= dag(U)
    HR[k] *= dag(V)
    HR[k + 1] *= V
    G[k]*=dag(V)
  end
  return HL, HR, Ss, G, ss 
end

function truncateG(G::ITensor, igl::Index; cutoff=1e-15, kwargs...)
  @mpoc_assert order(G) == 2
  igr = noncommonind(G, igl)
  @mpoc_assert tags(igl) != tags(igr) || plev(igl) != plev(igr) #Make sure subtensr can distinguish igl and igr
  M = G[igl => 2:(dim(igl) - 1), igr => 2:(dim(igr) - 1)]
  iml, = inds(M; plev=plev(igl)) #tags are the same, so plev is the only way to distinguish.
  U, s, V, spectrum, iu, iv = svd(M, iml; cutoff=cutoff,kwargs...)
  #
  # Build up U+, S+ and V+
  #
  iup = redim(iu, 1, 1, space(igl)) #Use redim to preserve QNs
  ivp = redim(iv, 1, 1, space(igr))
  #@show iu iup iv ivp igl s dense(s) U
  Up = grow(noprime(U), noprime(igl), dag(iup))
  Sp = grow(s, iup, ivp)
  Vp = grow(noprime(V), dag(ivp), noprime(igr))
  #
  #  But external link tags in so contractions with W[n] tensors will work.
  #
  Up=replacetags(Up, tags(iu), tags(igl))
  Sp=replacetags(Sp, tags(iu), tags(igl))
  Sp=replacetags(Sp, tags(iv), tags(igr))
  Vp=replacetags(Vp, tags(iv), tags(igr))
  #@mpoc_assert norm(dense(noprime(G))-dense(Up)*Sp*dense(Vp))<1e-12    #expensive!!!
  return Up, Sp, Vp, spectrum
end


