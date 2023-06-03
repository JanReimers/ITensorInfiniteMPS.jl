using SparseArrays

#-----------------------------------------------------------------------
#
#  Infinite lattice with unit cell
#
function gauge_fix!(H::reg_form_iMPO;kwargs...)
  @mpoc_assert H.ul==lower
  if !is_gauge_fixed(H;kwargs...)
    Wbs=extract_blocks(H,left; Abcd=true,fix_inds=true,swap_bc=true)
    sₙ, tₙ = Solve_b0c0(H,Wbs)
    for n in eachindex(H)
      gauge_fix!(H[n],sₙ[n - 1], sₙ[n],tₙ[n - 1], tₙ[n],Wbs[n])
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

function Solve_b0c0(Hrf::reg_form_iMPO,Wbs::CelledVector{regform_blocks})
  @assert length(Hrf)==length(Wbs)
  @assert translator(Hrf)==translator(Wbs)
  @assert id(Wbs[1].𝐀̂.ileft)==id(Wbs[end].𝐀̂.iright) #make sure periodic links were set up properly.
  N=length(Wbs)
  A0s = CelledVector{Matrix{Float64}}(undef,N)
  b0s = Vector{Float64}()
  c0s = Vector{Float64}()
  combiner_right_s=Vector{ITensor}() #Combiners
  i_b_right_column_s=Vector{Index}() #extra dim=1 indices on b and c
  n, nr, nc = 0, 0, 0
  irb, icb = CelledVector{Int64}(undef,N), CelledVector{Int64}(undef,N)
  ir, ic = 1, 1
  for (W,Wb) in zip(Hrf,Wbs)
    check(W)
    n+=1
    cl,cr=combiner(Wb.𝐀̂.ileft;tags="cl,ir=$ir"),combiner(Wb.𝐀̂.iright;tags="cr,ic=$ic")
    icl,icr=combinedind(cl),combinedind(cr)
    A_0=A0(Wb)*cl*cr #Project the 𝕀 subspace, should be just one block.
    # push!(A0s,sparse(matrix(icl, A_0, icr)))
    A0s[n]=matrix(icl, A_0, icr)
    b_0=b0(Wb)*cl
    c_0=c0(Wb)*cr
    append!(b0s, vector_o2(b_0))
    append!(c0s, vector_o2(c_0))
    irb[n]=ir
    icb[n]=ic
    # push!(irb, ir)
    # push!(icb, ic)
    push!(combiner_right_s,cr)
    push!(i_b_right_column_s,Wb.𝐛̂.iright)
    
    nr += size(A_0, 1)
    nc += size(A_0, 2)
    ir += size(A_0, 1)
    ic += size(A_0, 2)
    # @show nr nc ir ic
  end
  @assert nr == nc
  nn = nr
  # N = length(A0s)
  Ms, Mt = spzeros(nn, nn), spzeros(nn, nn)
  # @show N nn irb icb
  irs,irt,ic=1,1,1 #Cumulative row and col indexes
  for n in eachindex(A0s)
    nr, nc = size(A0s[n])
    nrp,ncp=size(A0s[n+1])
    irp= irs+nr
    icp= ic+nc
    # @show nr nc irs irt ic irp icp
    if irp>nn
      irp-=nn
    end
    if icp>nn
      icp-=nn
    end
    #
    #  These system will generally not bee so big that sparse improves performance significantly.
    #
    #droptol!(A0s[n], 1e-15)

    # @show irp icp 
    Ms[irs:irs+nr - 1, ic:ic+nc- 1] = A0s[n]
    Ms[irp:irp+nrp-1, ic:ic+nc- 1] -= sparse(LinearAlgebra.I, nrp, nc) #For Ncell=1 they overlap, hence we use -=
    Mt[irt:irt+nrp - 1, ic:ic+nc- 1] = sparse(LinearAlgebra.I, nrp, nc)
    Mt[irt:irt+nrp-1, icp:icp+ncp- 1] -= A0s[n+1] #For Ncell=1 they overlap, hence we use -=

    irs+=size(A0s[n],1)
    irt+=size(A0s[n+1],1)
    ic+=size(A0s[n],2)
  end
  # display(Array(Ms))
  # display(Array(Mt))
  s = Ms \ b0s
  t = Array(Base.transpose(Base.transpose(Mt) \ c0s))
  @assert norm(Ms * s - b0s) < 1e-15 * n
  @assert norm(Base.transpose(t * Mt) - c0s) < 1e-15 * n
  @assert size(t,1)==1 #t ends up as a 1xN matrix becuase of all the transposing.
  ss,ts=Vector{ITensor}(),Vector{ITensor}()
  irs,irt,ic=1,1,1 #Cumulative row and col indexes
  for n in 1:N
    nr, nc = size(A0s[n])
    sn=s[ic:ic+nc - 1]
    tn=t[1,ic:ic + nc - 1]
    icr=combinedind(combiner_right_s[n]) 
    @assert dim(i_b_right_column_s[n])==1
    # @show n sn icr icrp
    snT=ITensor(Float64,sn,icr,dag(i_b_right_column_s[n]))
    tnT=ITensor(Float64,tn,icr,dag(i_b_right_column_s[n]))
    snT=dag(snT*dag(combiner_right_s[n]))
    tnT=tnT*dag(combiner_right_s[n])
    # @show inds(snT) inds(tnT)
    push!(ss,snT)
    push!(ts,tnT)
    irs+=size(A0s[n],1)
    irt+=size(A0s[n+1],1)
    ic+=size(A0s[n],2)
  end
  return CelledVector(ss,translator(Hrf)),CelledVector(ts,translator(Hrf))
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
