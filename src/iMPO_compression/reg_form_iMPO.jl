#-----------------------------------------------------------------------
#
#  Infinite lattice iMPO with repeating unit cell
#
mutable struct reg_form_iMPO <: AbstractInfiniteMPS
  data::CelledVector{reg_form_Op}
  reverse::Bool
  ul::reg_form
  function reg_form_iMPO(H::InfiniteMPO, ul::reg_form)
    N = length(H)
    data = CelledVector{reg_form_Op}(undef, N,translator(H))
    for n in eachindex(H)
      il, ir = commonind(H[n],H[n-1]),commonind(H[n],H[n+1])
      data[n] = reg_form_Op(H[n], il, ir, ul)
    end
    return new(data, false, ul)
  end
  function reg_form_iMPO(
    Ws::CelledVector{reg_form_Op}, reverse::Bool, ul::reg_form
  )
    return new(Ws, reverse, ul)
  end
end

  function ITensorInfiniteMPS.translatecell(
    translator::Function, Wrf::reg_form_Op, n::Integer
  )
    new_inds = ITensorInfiniteMPS.translatecell(translator, inds(Wrf), n)
    W = setinds(Wrf.W, new_inds)
    ileft, iright = parse_links(W)
    return reg_form_Op(W, ileft, iright, Wrf.ul)
  end

  function translatecell(tf::Function, Wrb::regform_blocks, n::Integer)
    @assert !isnothing(Wrb.𝐀̂)
    @assert !isnothing(Wrb.𝐛̂)
    @assert !isnothing(Wrb.𝐜̂)
    @assert !isnothing(Wrb.𝐝̂)
    𝕀=translatecell(tf,Wrb.𝕀,n)
    A=translatecell(tf,Wrb.𝐀̂,n)
    b=translatecell(tf,Wrb.𝐛̂,n)
    c=translatecell(tf,Wrb.𝐜̂,n)
    d=translatecell(tf,Wrb.𝐝̂,n)
    Ac= isnothing(Wrb.𝐀̂𝐜̂) ? nothing : translatecell(tf,Wrb.𝐀̂𝐜̂,n)
    V= isnothing(Wrb.𝐕̂) ? nothing : translatecell(tf,Wrb.𝐕̂,n)
    return regform_blocks(𝕀,A,b,c,d,Ac,V)
  end
  
  translator(H::reg_form_iMPO)=translator(data(H))

  data(H::reg_form_iMPO) = H.data
  
  function Ws(H::reg_form_iMPO)
    return map(n -> H[n].W, 1:length(H))
  end

  function fix_inds(Wb1::regform_blocks,Wb2::regform_blocks)
    Wb1.𝐝̂=replaceind(Wb1.𝐝̂, Wb1.𝐝̂.iright, settags(dag(Wb2.𝐝̂.ileft),tags(Wb1.𝐝̂.iright)))
    # Wb1.icd=settags(dag(Wb2.𝐝̂.ileft),tags(Wb1.𝐝̂.iright))
    Wb1.𝐀̂=replaceind(Wb1.𝐀̂, Wb1.𝐀̂.iright, settags(dag(Wb2.𝐀̂.ileft),tags(Wb1.𝐀̂.iright)) )
    # Wb1.icA=settags(dag(Wb2.irA),tags(Wb1.icA))

    Wb1.𝐜̂=replaceind(Wb1.𝐜̂, Wb1.𝐜̂.iright, Wb1.𝐀̂.iright)
    # Wb1.icc=Wb1.icA
    Wb1.𝐜̂=replaceind(Wb1.𝐜̂, Wb1.𝐜̂.ileft, Wb1.𝐝̂.ileft)
    # Wb1.irc=Wb1.ird
    Wb1.𝐛̂=replaceind(Wb1.𝐛̂, Wb1.𝐛̂.ileft, Wb1.𝐀̂.ileft)
    # Wb1.irb=Wb1.irA
    Wb1.𝐛̂=replaceind(Wb1.𝐛̂, Wb1.𝐛̂.iright, Wb1.𝐝̂.iright)
    # Wb1.icb=Wb1.icd
    # check(reg_form_Op( Wb1.𝐀̂, Wb1.irA, Wb1.icA,lower))
    @assert id( Wb1.𝐀̂.iright)==id( Wb2.𝐀̂.ileft)
    return Wb1
  end

  function ITensorMPOCompression.extract_blocks(H::reg_form_iMPO,lr::orth_type;kwargs...)
    Wbsv=ITensorMPOCompression.extract_blocks.(H,lr;kwargs...)
    Wbs=CelledVector(Wbsv,translator(H))
    N=length(Wbs)
    for n in 1:N
      Wbs[n]=fix_inds(Wbs[n],Wbs[n+1])
      # check(reg_form_Op(Wbs[n].𝐀̂,Wbs[n].irA,Wbs[n].icA,lower))
      @assert id(Wbs[n].𝐀̂.iright)==id(Wbs[n+1].𝐀̂.ileft)
    end
    return Wbs
  end
  
  Base.length(H::reg_form_iMPO) = length(H.data)
  function Base.reverse(H::reg_form_iMPO)
    return reg_form_iMPO(Base.reverse(H.data),H.reverse, H.ul)
  end
  Base.iterate(H::reg_form_iMPO, args...) = iterate(H.data, args...)
  Base.getindex(H::reg_form_iMPO, n::Integer) = getindex(H.data, n)
  Base.setindex!(H::reg_form_iMPO, W::reg_form_Op, n::Integer) = setindex!(H.data, W, n)
  Base.copy(H::reg_form_iMPO) = reg_form_iMPO(copy(H.data), H.reverse, H.ul)
  
  function reg_form_iMPO(H::InfiniteMPO;honour_upper=false, kwargs...)
    (bl, bu) = detect_regular_form(H;kwargs...)
    if !(bl || bu)
      throw(ErrorException("MPO++(H::MPO), H must be in either lower or upper regular form"))
    end
    if (bl && bu)
      @pprint(H[1])
      @assert false
    end
    ul::reg_form = bl ? lower : upper #if both bl and bu are true then something is seriously wrong
    Hrf=reg_form_iMPO(H, ul)
    if ul==upper && !honour_upper
      Hrf=transpose(Hrf)
    end
    check(Hrf)
    return Hrf
  end
  
  #same as reg_form_MPO version
  function ITensorMPOCompression.check(Hrf::reg_form_iMPO)
    check.(Hrf)
  end

  function ITensorInfiniteMPS.InfiniteMPO(Hrf::reg_form_iMPO)::InfiniteMPO
    return InfiniteMPO(Ws(Hrf))
  end
  
  
  function ITensorMPOCompression.get_Dw(Hrf::reg_form_iMPO)
    return map(n -> dim(Hrf[n].iright), eachindex(Hrf))
  end
  
  ITensors.maxlinkdim(Hrf::reg_form_iMPO)=maximum(get_Dw(Hrf))
  ITensors.maxlinkdim(Hrf::reg_form_MPO)=maximum(get_Dw(Hrf))
  
  ITensorMPOCompression.check_ortho(H::InfiniteMPO,lr::orth_type)=check_ortho(reg_form_iMPO(H),lr)
  function ITensors.linkind(M::AbstractInfiniteMPS, j::Integer)
    return commonind(M[j], M[j + 1])
  end

  function detect_regular_form(H::AbstractInfiniteMPS;kwargs...)::Tuple{Bool,Bool}
    return is_regular_form(H, lower;kwargs...), is_regular_form(H, upper;kwargs...)
  end

  function is_regular_form(H::AbstractInfiniteMPS, ul::reg_form;kwargs...)::Bool
    il = dag(linkind(H, 0))
    for n in 1:length(H)
      ir = linkind(H, n)
      #@show il ir inds(H[n])
      Wrf = reg_form_Op(H[n], il, ir, ul)
      !ITensorMPOCompression.is_regular_form(Wrf;kwargs...) && return false
      il = dag(ir)
    end
    return true
  end
  function get_Dw(H::InfiniteMPO)::Vector{Int64}
    N = length(H)
    Dws = Vector{Int64}(undef, N)
    for n in 1:N
      l = commonind(H[n], H[n + 1])
      Dws[n] = dim(l)
    end
    return Dws
  end  
  #
  #  Handles direction only.  For iMPOs we include the last site in the unit cell.
  #
  function sweep(H::AbstractInfiniteMPS, lr::orth_type)::StepRange{Int64,Int64}
    N = length(H)
    return lr == left ? (1:1:N) : (N:-1:1)
  end

  function ITensorMPOCompression.check_ortho(H::reg_form_iMPO, lr::orth_type;kwargs...)::Bool
    for n in sweep(H, lr) #skip the edge row/col opertors
      !check_ortho(H[n], lr;kwargs...) && return false
    end
    return true
  end
  
  function is_regular_form(H::reg_form_iMPO;kwargs...)::Bool
    for W in H
      !is_regular_form(W;kwargs...) && return false
    end
    return true
  end

  function Base.transpose(Hrf::reg_form_iMPO)::reg_form_iMPO
    Ws=copy(data(Hrf))
    N=length(Hrf)
    ul1=mirror(Hrf.ul)
    for n in 1:N
      ir=Ws[n].iright
      G = G_transpose(ir, reverse(ir))
      Ws[n]*=G
      Ws[n+1]*=dag(G)
      Ws[n]=reg_form_Op(Ws[n].W,Ws[n].ileft,Ws[n].iright,ul1)
      # Ws[n].ul=ul1 setting member data in CelledVector is tricky because of the translator.
    end
    return reg_form_iMPO(Ws, false, ul1)
  end
  