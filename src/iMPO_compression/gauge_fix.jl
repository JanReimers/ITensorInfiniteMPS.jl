using SparseArrays
import ITensorMPOCompression: is_gauge_fixed, extract_blocks, A0, b0, c0, vector_o2,  
      set_𝐛̂_block!, set_𝐜̂_block!, set_𝐝̂_block!

#-----------------------------------------------------------------------
#
#  Infinite lattice with unit cell
#
function gauge_fix!(H::reg_form_iMPO;kwargs...)
  @mpoc_assert H.ul==lower
  if !is_gauge_fixed(H;kwargs...)
    Wbs=extract_blocks(H,left; all=true,fix_inds=true)
    sₙ, tₙ = Solve_b0c0(H,Wbs)
    for n in eachindex(H)
      gauge_fix!(H[n],sₙ[n - 1], sₙ[n],tₙ[n - 1], tₙ[n],Wbs[n])
    end
  end
end

function ITensorInfiniteMPS.translatecell(::Function, T::Float64, ::Integer)
  return T
end

function Solve_b0c0(Hrf::reg_form_iMPO,Wbs::Vector{regform_blocks})
  @assert length(Hrf)==length(Wbs)
  A0s = Vector{Matrix}()
  b0s = Vector{Float64}()
  c0s = Vector{Float64}()
  combiner_right_s=Vector{ITensor}() #Combiners
  i_b_right_column_s=Vector{Index}() #extra dim=1 indices on b and c
  nr, nc = 0, 0
  irb, icb = Vector{Int64}(), Vector{Int64}()
  ir, ic = 1, 1
  for (W,Wb) in zip(Hrf,Wbs)
    check(W)
    cl,cr=combiner(Wb.irA;tags="cl,ir=$ir"),combiner(Wb.icA;tags="cr,ic=$ic")
    icl,icr=combinedind(cl),combinedind(cr)
    A_0=A0(Wb)*cl*cr #Project the 𝕀 subspace, should be just one block.
    push!(A0s,sparse(matrix(icl, A_0, icr)))
    b_0=b0(Wb)*cl
    c_0=c0(Wb)*cr
    append!(b0s, vector_o2(b_0))
    append!(c0s, vector_o2(c_0))
    push!(irb, ir)
    push!(icb, ic)
    push!(combiner_right_s,cr)
    push!(i_b_right_column_s,Wb.icb)
    
    nr += size(A_0, 1)
    nc += size(A_0, 2)
    ir += size(A_0, 1)
    ic += size(A_0, 2)
  end
  @assert nr == nc
  n = nr
  N = length(A0s)
  Ms, Mt = spzeros(n, n), spzeros(n, n)
  for n in eachindex(A0s)
    nr, nc = size(A0s[n])
    ir, ic = irb[n], icb[n]
    #
    #  These system will generally not bee so big that sparse improves performance significantly.
    #
    #droptol!(A0s[n], 1e-15)
    Id=sparse(LinearAlgebra.I, nr, nc)
    Ms[ir:(ir + nr - 1), ic:(ic + nc - 1)] = A0s[n]
    Mt[ir:(ir + nr - 1), ic:(ic + nc - 1)] = Id
    if n == 1
      Ms[ir:(ir + nr - 1), icb[N]:(icb[N] + nc - 1)] -= Id
      Mt[ir:(ir + nr - 1), icb[N]:(icb[N] + nc - 1)] -= A0s[n]
    else
      Ms[ir:(ir + nr - 1), icb[n - 1]:(ic - 1)] -= Id
      Mt[ir:(ir + nr - 1), icb[n - 1]:(ic - 1)] -= A0s[n]
    end
  end
  s = Ms \ b0s
  t = Array(Base.transpose(Base.transpose(Mt) \ c0s))
  @assert norm(Ms * s - b0s) < 1e-15 * n
  @assert norm(Base.transpose(t * Mt) - c0s) < 1e-15 * n
  @assert size(t,1)==1 #t ends up as a 1xN matrix becuase of all the transposing.
  ss,ts=Vector{ITensor}(),Vector{ITensor}()
  for n in 1:N
    sn=s[irb[n]:(irb[n] + nr - 1)]
    tn=t[1,irb[n]:(irb[n] + nr - 1)]
    icr=combinedind(combiner_right_s[n])
    @assert dim(i_b_right_column_s[n])==1
    snT=ITensor(Float64,sn,icr,dag(i_b_right_column_s[n]))
    tnT=ITensor(Float64,tn,icr,dag(i_b_right_column_s[n]))
    snT=dag(snT*dag(combiner_right_s[n]))
    tnT=tnT*dag(combiner_right_s[n])
    push!(ss,snT)
    push!(ts,tnT)
  end
  return CelledVector(ss), CelledVector(ts)
end

function gauge_fix!(
  W::reg_form_Op,
  𝒔ₙ₋₁::ITensor,
  𝒔ₙ::ITensor,
  𝒕ₙ₋₁::ITensor,
  𝒕ₙ::ITensor,
  Wb::regform_blocks
)
  @assert is_regular_form(W)
  #Wb = extract_blocks(W, left; all=true, fix_inds=true)
  𝕀, 𝐀̂, 𝐛̂, 𝐜̂, 𝐝̂ = Wb.𝕀, Wb.𝐀̂, Wb.𝐛̂, Wb.𝐜̂, Wb.𝐝̂ #for readability below.
  irs=dag(noncommonind(𝒔ₙ₋₁,Wb.irb))
  ict=dag(noncommonind(𝒕ₙ,Wb.icc))

  𝐛̂⎖ = 𝐛̂ + 𝒔ₙ₋₁ * 𝕀 *δ(Wb.icb,irs) -  𝐀̂ * 𝒔ₙ
  𝐜̂⎖ = 𝐜̂ - 𝒕ₙ * 𝕀 *δ(Wb.irc,ict) +  𝒕ₙ₋₁ * 𝐀̂
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
