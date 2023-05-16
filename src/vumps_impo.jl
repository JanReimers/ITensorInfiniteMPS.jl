import ITensorMPOCompression: slice, assign!

struct ALk
    ψ::InfiniteCanonicalMPS
    k::Int
end

function (A::ALk)(x)
    ψ = A.ψ
    ψ′ =dag(ψ)'
    k = A.k
    N = nsites(ψ)
    l = linkinds(only, ψ.AL)
    # l′ = linkinds(only, ψ′.AL)
    r = linkinds(only, ψ.AR)
    # r′ = linkinds(only, ψ′.AR)
    s = siteinds(only, ψ)
    δʳ(kk) = δ(dag(r[kk]), prime(r[kk]))
    δˡ(kk) = δ(dag(l[kk]), prime(l[kk]))
    # δˡ(kk) = δ(l[kk], l′[kk])
    δˢ(kk) = δ(dag(s[kk]), prime(s[kk]))
    xT = translatecell(translator(ψ), x, -1) #xT = x shifted one unit cell to the left.
    for k′=k-N+1:k
        xT*=ψ′.AL[k′]*δˢ(k′)*ψ.AL[k′]
        # @show ψ′.AL[k′]*δˢ(k′)*ψ.AL[k′]
    end
    R=ψ.C[k] * (ψ′.C[k] * δʳ(k))
    𝕀=denseblocks(dag(δˡ(k)))
    xR=x*R*𝕀
    # @show xT xR
    return xT-xR
end

struct ARk
    ψ::InfiniteCanonicalMPS
    k::Int
end

function (A::ARk)(x)
    ψ = A.ψ
    ψ′ =dag(ψ)'
    k = A.k
    N = nsites(ψ)
    l = linkinds(only, ψ.AL)
    r = linkinds(only, ψ.AR)
    s = siteinds(only, ψ)
    δʳ(kk) = δ(dag(r[kk]), prime(r[kk]))
    δˡ(kk) = δ(dag(l[kk]), prime(l[kk]))
    δˢ(kk) = δ(dag(s[kk]), prime(s[kk]))

    xT = translatecell(translator(ψ), x, 1) #xT = x shifted one unit cell to the right.
    for k′=k+N:-1:k+1
        # @show k k′ inds(ψ.AR[k′])
        xT*=ψ′.AR[k′]*δˢ(k′)*ψ.AR[k′]
        @assert order(xT)==2
    end
    R=ψ.C[k] * (ψ′.C[k]*dag(δˡ(k)))
    𝕀=denseblocks(δʳ(k))
    xR=x*R*𝕀 # (x|R)(𝕀|
    return xT-xR
end

#
#   (Lₖ|=(Lₖ₋₁|*T(k)ᵂₗ
#
function apply_TW_left(Lₖ₋₁::ITensor,Ŵ::ITensor,ψ::ITensor)
    return dag(ψ)'*(Lₖ₋₁*Ŵ)*ψ #parentheses may affect perfomance.
end
#
#   |Rₖ₋₁)=T(k)ᵂᵣ*|Rₖ) 
#     This is semantically identical to the left version
#     The only reason to keep separate versions is so that the variable name Rₖ makes it 
#     totally obvious what we are doing.
#
function apply_TW_right(Rₖ::ITensor,Ŵ::ITensor,ψ::ITensor)
    return dag(ψ)'*(Ŵ*Rₖ)*ψ #parentheses may affect perfomance.
end

# solve:
#
#    /------ψₖ--    /----
#    |      |       |
#    Lᵂₖ₋₁--Hₖ-- == Lᵂₖ--
#    |      |       |
#    \------ψₖ'-    \----
#

function left_environment(H::InfiniteMPO, ψ::InfiniteCanonicalMPS; tol=1e-10)
    l = linkinds(only, ψ.AL)
    r = linkinds(only, ψ.AR)
    lₕ= linkinds(only, H)
    δʳ(kk) = δ(dag(r[kk]), prime(r[kk]))
    δˡ(kk) = δ(dag(l[kk]), prime(l[kk]))
    D=dim(l[1])
    N=nsites(ψ)
    il =lₕ[1] #We want the right link of H[1]
    Dw=dim(il)
    #
    #  Solve for k=1 using the unit cell transfer matrix 𝕋ᵂ
    #
    L₁=ITensor(0.0,l[1],dag(l[1]'),il)
    assign!(L₁,denseblocks(δˡ(1)),il=>Dw) #Left eigen vector of TL (not TWL)
    for b in Dw-1:-1:1 #sweep backwards from Dw-1 down.
        Lₖ₋₁=translatecell(translator(ψ), L₁, -1) #  Load up all the known tensors from b1+1 to Dw.  Also translate one unit to the left.
        for k in 2-N:1 #  Loop throught the unit cell and apply Tᵂₗ
            Lₖ₋₁=apply_TW_left(Lₖ₋₁,H[k],ψ.AL[k])
            @assert order(Lₖ₋₁)==3
        end # for k
        assign!(L₁, slice(Lₖ₋₁,il=>b),il=>b)  #save the new value.
        L₁[il=>b:b]=Lₖ₋₁[il=>b:b]
    end #for b
    #
    #  At this point 
    #    1) we have solved for all of L₁ except the first element, where the inversion must be solved.
    #    2) L₁ = YL₁ which is the inhomogenious part of the equation L₁=L₁*𝕋ᵂ₁+YL₁
    #    3) We also need to subtract out the extensive portion 𝕋ᵂ₁₁: |R)(𝕀| which happens inside linsolve
    #    4) Apparently we also need to correct YL₁ in the same way  YL₁ = YL₁ - YL₁*|R)(𝕀|
    #
    YL₁=slice(L₁,il=>1) #Pluck out the first element as YL₁
    R = ψ.C[1] * δʳ(1) * dag(ψ.C[1]')  # |R)
    𝕀 = denseblocks(dag(δˡ(1))) # (𝕀|
    eₗ = scalar(YL₁ * R) #Get energy for whole unit cell ... before  YL₁ get modified
    YL₁ = YL₁ - eₗ * 𝕀 #from Loic's MPOMatrix code.
    A = ALk(ψ, 1)
    L₁₁, info = linsolve(A, YL₁, 1, -1; tol=tol)
    assign!(L₁,L₁₁,il=>1)
    #
    #  Now sweep throught the cell and evalute all the other L[k] form L[1]
    #
    L=CelledVector{ITensor}(undef,N)
    L[1]=L₁
    for k in 2:N
        L[k]=apply_TW_left(L[k-1],H[k],ψ.AL[k])
    end
    #
    #  Verify that we get L[1] back from L[0].  TODO: RIp this out later once we gain confidence
    #
    L₁=apply_TW_left(L[0],H[1],ψ.AL[1])
    pass=true
    for b in 2:Dw #We know that L₁[1] is wrong
        L₁b=slice(L₁,il=>b)
        L1b=slice(L[1],il=>b)
        if norm(L₁b-L1b)>1e-14*D*N
            @show b L₁b L1b
            pass= false
        end
    end
    @assert pass
    
    # assign!(L[1],slice(L₁,il=>1),il=>1)

    return L,eₗ
end

# solve:
#
#    --ψₖ--\       --\
#      |    |         |
#    --Hₖ---Rᵂₖ == ---Rᵂₖ₋₁
#      |    |         |
#    --ψₖ'-/       --/
#
function right_environment(H::InfiniteMPO, ψ::InfiniteCanonicalMPS; tol=1e-10)
    l = linkinds(only, ψ.AL)
    r = linkinds(only, ψ.AR)
    lₕ= linkinds(only, H)
    δʳ(kk) = δ(dag(r[kk]), prime(r[kk]))
    δˡ(kk) = δ(dag(l[kk]), prime(l[kk]))
    D=dim(l[1])
    ir=dag(lₕ[1]) #left link of H[2] 
    Dw=dim(ir)
    N=nsites(ψ)
    #
    #  Solve for k=1 using the unit cell transfer matrix 𝕋ᵂ
    #
    R₁=ITensor(0.0,dag(r[1]),r[1]',ir)
    assign!(R₁,denseblocks(δʳ(1)),ir=>1) #right eigen vector of TR 
    for b in 2:Dw 
        Rₖ=translatecell(translator(ψ), R₁, 1) #Load up all the know tensors from 1 to b1-1.  Also translate one unit to the right.
        for k in N+1:-1:2 #  Loop throught the unit cell and apply Tᵂₗ
            Rₖ=apply_TW_right(Rₖ,H[k],ψ.AR[k])
            @assert order(Rₖ)==3
        end # for k
        # assign!(R₁,slice(Rₖ,ir=>b),ir=>b)  #save the new value.
        R₁[ir=>b:b]=Rₖ[ir=>b:b]
    end #for b1
    #
    #  See comments above in the same section of the left_environment function
    #
    YR_Dw=slice(R₁,ir=>Dw)   
    L = ψ.C[1] * dag(δˡ(1)) * dag(ψ.C[1]') # (L|
    𝕀 =  denseblocks(δʳ(1)) # |𝕀)
    eᵣ = scalar(L*YR_Dw) #Get energy for whole unit cell ... before  YL₁ get modified
    YR_Dw = YR_Dw - 𝕀 * eᵣ
    A = ARk(ψ, 1)
    R₁Dw, info = linsolve(A, YR_Dw, 1, -1; tol=tol)
    assign!(R₁,R₁Dw,ir=>Dw)
    #
    #  Now sweep leftwards through the cell and evalaute all the R[k] form R[1]
    #
    R=CelledVector{ITensor}(undef,N)
    R[1]=R₁
    for k in N:-1:2
        R[k]=apply_TW_right(R[k+1],H[k+1],ψ.AR[k+1])
    end
    #
    #  Verify that we get R[1] back from R[2].  TODO: Scrap this once we gain confidence.
    #
    R₁=apply_TW_right(R[2],H[2],ψ.AR[2])
    for b in 1:Dw-1 #We know that R₁[Dw] is wrong
        R₁b=slice(R₁,ir=>b)
        R1b=slice(R[1],ir=>b)
        if norm(R₁b-R1b)>1e-14*D*N
            @show b R₁b R1b
            @assert  false
        end
    end
   
    # assign!(R[1],slice(R₁,ir=>Dw),ir=>Dw)

    return R,eᵣ
end

struct iMPO⁰
    L::ITensor
    R::ITensor
end

H⁰(L::ITensor,R::ITensor)=iMPO⁰(L,R)

function (H::iMPO⁰)(x)
    return noprime(H.L*x*H.R)
end

struct iMPO¹
    L::ITensor
    R::ITensor
    Ŵ::ITensor
  end
  
function (H::iMPO¹)(x)
    return noprime(H.L*(x*H.Ŵ)*H.R)
end

H¹(L::ITensor,R::ITensor,Ŵ::ITensor)=iMPO¹(L,R,Ŵ)

