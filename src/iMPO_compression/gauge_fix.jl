using SparseArrays

#-----------------------------------------------------------------------
#
#  Infinite lattice with unit cell
#
function ITensorMPOCompression.gauge_fix!(H::reg_form_iMPO;kwargs...)
  @mpoc_assert H.ul==lower
  if !is_gauge_fixed(H;kwargs...)
    Wbs=extract_blocks(H,left; Abcd=true,fix_inds=true,swap_bc=true)
    sₙ, tₙ = Solve_b0c0(H,Wbs)
    for n in eachindex(H)
      gauge_fix_Op!(H[n],sₙ[n - 1], sₙ[n],tₙ[n - 1], tₙ[n],Wbs[n])
    end
  end
end

function ITensorInfiniteMPS.translatecell(::Function, T::Float64, ::Integer)
  return T
end

function ITensorInfiniteMPS.translatecell(::Function, m::Matrix{Float64}, ::Integer)
  return m
end
function ITensorInfiniteMPS.translatecell(::Function, i::Int64, ::Integer)
  return i
end
function ITensorInfiniteMPS.translatecell(::Function, ur::UnitRange{Int64}, ::Integer)
  return ur
end

#
#  This code implements the algo decribed in the section "3.1 Gauge fixing for Ac-block respecting decomposition"
#  in the document  TechnicalDetails.pdf
#
function Solve_b0c0(Hrf::reg_form_iMPO,Wbs::CelledVector{regform_blocks})
  @assert length(Hrf)==length(Wbs)
  @assert translator(Hrf)==translator(Wbs)
  @assert id(Wbs[1].𝐀̂.ileft)==id(Wbs[end].𝐀̂.iright) #make sure periodic links were set up properly.
  N=length(Wbs)
  A0s = CelledVector{Matrix{Float64}}(undef,N)
  b0s = Vector{Float64}()
  c0s = Vector{Float64}()
  rrs= CelledVector{UnitRange{Int64}}(undef,N) #row ranges
  crs= CelledVector{UnitRange{Int64}}(undef,N) #col ranges
  right_combiners=Vector{ITensor}() #Combiners
  right_indices=Vector{Index}() #extra dim=1 indices on b and c
  n, nnr, nnc = 0, 0, 0
  ir,ic=1,1
  for (W,Wb) in zip(Hrf,Wbs)
    check(W)
    n+=1
    cl,cr=combiner(Wb.𝐀̂.ileft;tags="cl,n=$n"),combiner(Wb.𝐀̂.iright;tags="cr,n=$n")
    icl,icr=combinedind(cl),combinedind(cr)
    #The combiners should project out the I block.  All other blocks will be zero.
    A0s[n]=matrix(icl,A0(Wb)*cl*cr, icr)
    append!(b0s, vector_o2(b0(Wb)*cl))
    append!(c0s, vector_o2(c0(Wb)*cr))
    # These are used to rebuild ITensors below.
    push!(right_combiners,cr)
    push!(right_indices,Wb.𝐛̂.iright)
    #
    #  Track row/column blocks and total number of rows/cols.
    #
    nr,nc=size(A0s[n])
    nnr += nr
    nnc += nc
    rrs[n]=ir:ir+nr-1
    crs[n]=ic:ic+nc-1
    ir+=nr
    ic+=nc
  end
  @assert nnr == nnc #individual site A0s are not nessecarily square, but the whole/unit-cell system should be
  nn = nnr
  #
  #  Fill out the Ms and Mt matrices
  #
  Ms, Mt = spzeros(nn, nn), spzeros(nn, nn)
  for n in eachindex(A0s)
    nc = size(A0s[n],2)
    nrp=size(A0s[n+1],1)
    # Ms and Mt have different row blocks.  Mt's row blocks are shifted by one block.
    rrst= rrs[n+1].-rrs[1].stop
    rrst= n<N ? rrst : rrst.+nn
    Ms[rrs[n], crs[n]] = sparse(A0s[n])
    Ms[rrs[n+1],crs[n]] -= sparse(LinearAlgebra.I, nrp, nc) #For Ncell=1 they overlap, hence we use -=
    Mt[rrst, crs[n]] = sparse(LinearAlgebra.I, nrp, nc)
    Mt[rrst, crs[n+1]] -= sparse(A0s[n+1]) #For Ncell=1 they overlap, hence we use -=
  end
  # display(Array(Ms))
  # display(Array(Mt))
  s = Ms \ b0s
  t = Array(Base.transpose(Base.transpose(Mt) \ c0s))
  @assert norm(Ms * s - b0s) < 1e-15 * n
  @assert norm(Base.transpose(t * Mt) - c0s) < 1e-15 * n
  @assert size(t,1)==1 #t ends up as a 1xN matrix because of all the transposing.
  
  ss,ts=CelledVector{ITensor}(undef,N,translator(Hrf)),CelledVector{ITensor}(undef,N,translator(Hrf))
  for n in 1:N
    # make suitable indices
    cr=right_combiners[n]
    il1=combinedind(cr) 
    il2=dag(right_indices[n])
    @assert dim(il2)==1
    # Get data into an ITensor
    snT=ITensor(Float64,s[crs[n]],il1,il2)
    tnT=ITensor(Float64,t[1,crs[n]],il1,il2)
    # undo the combine operation and save.
    ss[n]=dag(snT*dag(cr))
    ts[n]=tnT*dag(cr)
  end
  return ss,ts
end

function gauge_fix_Op!(
  W::reg_form_Op,
  𝒔ₙ₋₁::ITensor,
  𝒔ₙ::ITensor,
  𝒕ₙ₋₁::ITensor,
  𝒕ₙ::ITensor,
  Wb::regform_blocks
)
  @assert is_regular_form(W)
  𝕀, 𝐀̂, 𝐛̂, 𝐜̂, 𝐝̂ = Wb.𝕀, Wb.𝐀̂.W, Wb.𝐛̂.W, Wb.𝐜̂.W, Wb.𝐝̂.W #for readability below.
  irs=dag(noncommonind(𝒔ₙ₋₁,Wb.𝐛̂.ileft))
  ict=dag(noncommonind(𝒕ₙ,Wb.𝐜̂.iright))
  
  𝐛̂⎖ = 𝐛̂ + 𝒔ₙ₋₁ * 𝕀 *δ(Wb.𝐛̂.iright,irs) -  𝐀̂ * 𝒔ₙ
  𝐜̂⎖ = 𝐜̂ - 𝒕ₙ * 𝕀 *δ(Wb.𝐜̂.ileft,ict) +  𝒕ₙ₋₁ * 𝐀̂
  𝐝̂⎖ = 𝐝̂ + 𝒕ₙ₋₁ * 𝐛̂ - 𝒔ₙ * 𝐜̂⎖
  
  set_𝐛̂_block!(W, 𝐛̂⎖)
  set_𝐜̂_block!(W, 𝐜̂⎖)
  set_𝐝̂_block!(W, 𝐝̂⎖)
  return check(W)
end
#
#  Make sure indices are ordered and then convert to a matrix
#
function NDTensors.matrix(il::Index, T::ITensor, ir::Index)
  T1 = ITensors.permute(T, il, ir; allow_alias=true)
  return matrix(T1)
end
