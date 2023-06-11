
function truncate!(
  Hi::reg_form_iMPO; rr_cutoff=1e-14, kwargs...
)::Tuple{reg_form_iMPO,reg_form_iMPO,Any,Any,bond_spectrums}
  #
  #  Orthogonalize and get the gauge transforms between HL and HR, and H0 and HR
  #
  HL,HR,GLR,G0R = orthogonalize(Hi;cutoff=rr_cutoff,kwargs...)
  #
  #  Now compress and gauge transforms and apply the similarity transform to HL and HR.
  #
  N = length(Hi)
  b_specs = bond_spectrums(undef, N)
  Ss = CelledVector{ITensor}(undef, N,translator(Hi))
  for k in 1:N
    #prime the right index of G so that indices can be distinguished when N==1.
    GLR[k] = prime(GLR[k],HR[k+1].ileft)
    U, Ss[k], V, b_specs[k] = truncateG(GLR[k], dag(HL[k].iright) ; kwargs...)
    HL[k] *= U #These contractions are overloaded in reg_form_Op to automatically align and update link indices.
    HL[k + 1] *= dag(U)
    HR[k] *= dag(V)
    HR[k + 1] *= V
    G0R[k]*=dag(V) #Update gauge transform between H0 and HR
  end
  return HL, HR, Ss, G0R, b_specs 
end

function truncateG(G::ITensor, igl::Index; cutoff=1e-15, kwargs...)
  @mpoc_assert order(G) == 2
  igr = noncommonind(G, igl)
  @mpoc_assert tags(igl) != tags(igr) || plev(igl) != plev(igr) #Make sure subtensor can distinguish igl and igr
  M = G[igl => 2:(dim(igl) - 1), igr => 2:(dim(igr) - 1)]
  iml, = inds(M; plev=plev(igl)) #tags are the same, so plev is the only way to distinguish.
  U, s, V, spectrum, iu, iv = svd(M, iml; cutoff=cutoff,kwargs...)
  #
  # Build up U+, S+ and V+
  #
  iup = redim(iu, 1, 1, space(igl)) #Use redim to preserve QNs
  ivp = redim(iv, 1, 1, space(igr))
  Up = grow(noprime(U), noprime(igl), dag(iup))
  Sp = grow(s, iup, ivp)
  Vp = grow(noprime(V), dag(ivp), noprime(igr))
  #
  #  Put external link tags back in so contractions with W[n] tensors will work.
  #
  Up=replacetags(Up, tags(iu), tags(igl))
  Sp=replacetags(Sp, tags(iu), tags(igl))
  Sp=replacetags(Sp, tags(iv), tags(igr))
  Vp=replacetags(Vp, tags(iv), tags(igr))
  return Up, Sp, Vp, spectrum
end


