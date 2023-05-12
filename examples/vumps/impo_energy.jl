using ITensors
using ITensorInfiniteMPS
using ITensorMPOCompression
using KrylovKit: linsolve
using Random 

Random.seed!(1234)

import ITensorInfiniteMPS: left_environment, right_environment, vumps_solver,tdvp_iteration_sequential
import ITensorInfiniteMPS: ALk, ARk
# , translatecell, vumps_solver,     tdvp_iteration_sequential,H⁰,H¹,ortho_polar
import ITensorMPOCompression: assign!, slice

#
#   (Lₖ|=(Lₖ₋₁|*T(k)ᵂₗ
#
function apply_TW_left1(Lₖ₋₁::ITensor,Ŵ::ITensor,ψ::ITensor)
    return dag(ψ)'*(Lₖ₋₁*Ŵ)*ψ #parentheses may affect perfomance.
end
#
#   |Rₖ₋₁)=T(k)ᵂᵣ*|Rₖ) 
#     This is semantically identical to the left version
#     The only reason to keep separate versions is so that the variable name Rₖ makes it 
#     totally obvious what we are doing.
#
function apply_TW_right1(Rₖ::ITensor,Ŵ::ITensor,ψ::ITensor)
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

function left_environment1(H::InfiniteMPO, ψ::InfiniteCanonicalMPS; tol=1e-10)
    l = linkinds(only, ψ.AL)
    r = linkinds(only, ψ.AR)
    lₕ= linkinds(only, H)
    δʳ(kk) = δ(dag(r[kk]), prime(r[kk]))
    δˡ(kk) = δ(dag(l[kk]), prime(l[kk]))
    D=dim(l[1])
    N=nsites(ψ)
    il =dag(lₕ[1]) #We want the left link of H[2]
    Dw=dim(il)
    #
    #  Solve for k=1 using the unit cell transfer matrix 𝕋ᵂ
    #
    L₁=ITensor(0.0,l[1],l[1]',il)
    assign!(L₁,denseblocks(δˡ(1)),il=>Dw) #Left eigen vector of TL (not TWL)
    for b in Dw-1:-1:1 #sweep backwards from Dw-1 down.
        Lₖ₋₁=translatecell(translator(ψ), L₁, -1) #  Load up all the known tensors from b1+1 to Dw.  Also translate one unit to the left.
        for k in 2-N:1 #  Loop throught the unit cell and apply Tᵂₗ
            Lₖ₋₁=apply_TW_left1(Lₖ₋₁,H[k],ψ.AL[k])
            @assert order(Lₖ₋₁)==3
        end # for k
        assign!(L₁,slice(Lₖ₋₁,il=>b),il=>b)  #save the new value.
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
    𝕀 = denseblocks(δˡ(1)) # (𝕀|
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
        L[k]=apply_TW_left1(L[k-1],H[k],ψ.AL[k])
    end
    #
    #  Verify that we get L[1] back from L[0].  TODO: RIp this out later once we gain confidence
    #
    L₁=apply_TW_left1(L[0],H[1],ψ.AL[1])
    for b in 2:Dw #We know that L₁[1] is wrong
        L₁b=slice(L₁,il=>b)
        L1b=slice(L[1],il=>b)
        if norm(L₁b-L1b)>1e-15*D*N
            @show L₁b L1b
            @assert  false
        end
    end

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
function right_environment1(H::InfiniteMPO, ψ::InfiniteCanonicalMPS; tol=1e-10)
    l = linkinds(only, ψ.AL)
    r = linkinds(only, ψ.AR)
    lₕ= linkinds(only, H)
    δʳ(kk) = δ(dag(r[kk]), prime(r[kk]))
    δˡ(kk) = δ(dag(l[kk]), prime(l[kk]))
    D=dim(l[1])
    ir=lₕ[1] #right link of H[1] 
    Dw=dim(ir)
    N=nsites(ψ)
    #
    #  Solve for k=1 using the unit cell transfer matrix 𝕋ᵂ
    #
    R₁=ITensor(0.0,r[1],r[1]',ir)
    assign!(R₁,denseblocks(δʳ(1)),ir=>1) #right eigen vector of TR 
    for b1 in 2:Dw 
        Rₖ=translatecell(translator(ψ), R₁, 1) #Load up all the know tensors from 1 to b1-1.  Also translate one unit to the right.
        for k in N+1:-1:2 #  Loop throught the unit cell and apply Tᵂₗ
            Rₖ=apply_TW_right1(Rₖ,H[k],ψ.AR[k])
            @assert order(Rₖ)==3
        end # for k
        assign!(R₁,slice(Rₖ,ir=>b1),ir=>b1)  #save the new value.
    end #for b1
    #
    #  See comments above in the same section of the left_environment function
    #
    YR_Dw=slice(R₁,ir=>Dw)   
    L = ψ.C[1] * δˡ(1) * dag(ψ.C[1]') # (L|
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
        R[k]=apply_TW_right1(R[k+1],H[k+1],ψ.AR[k+1])
    end
    #
    #  Verify that we get R[1] back from R[2].  TODO: Scrap this once we gain confidence.
    #
    R₁=apply_TW_right1(R[2],H[2],ψ.AR[2])
    for b in 1:Dw-1 #We know that R₁[Dw] is wrong
        R₁b=slice(R₁,ir=>b)
        R1b=slice(R[1],ir=>b)
        if norm(R₁b-R1b)>1e-15*D*N
            @show R₁b R1b
            @assert  false
        end
    end

    return R,eᵣ
end

vumps_kwargs = (
      multisite_update_alg="sequential",
    #   multisite_update_alg="parallel",
      tol=1e-8,
      maxiter=50,
      outputlevel=0,
      time_step=-Inf,
    )





 
#
# tdvp_iteration tests
#

# let 
#     println("----------------------------------------------")
#     initstate(n) = isodd(n) ? "↑" : "↓"
#     N,nex=2,0
#     s = siteinds("S=1/2", N; conserve_qns=false)
#     si = infsiteinds(s)
#     ψ = InfMPS(si, initstate)
#     Hm = InfiniteMPOMatrix(Model("heisenberg"), si)
#     # ψ = tdvp(Hm, ψ; vumps_kwargs...)
#     # for _ in 1:nex
#     #     ψ = subspace_expansion(ψ, Hm; cutoff=1e-8,maxdim=16)
#     #     ψ = tdvp(Hm, ψ; vumps_kwargs...)
#     # end
#     ψ,(el,er)=ITensorInfiniteMPS.tdvp_iteration_sequential(vumps_solver,Hm,ψ;time_step=-Inf)
#     @show el er 

#     H = InfiniteMPO(Model("heisenberg"), si)
#     # ψ,(el,er)=tdvp_iteration_sequential(vumps_solver,H,ψ;time_step=-Inf)
#     # @show el er 
#     ψ = tdvp(H, ψ; vumps_kwargs...)
#     for i in 1:4
#         ψ = subspace_expansion(ψ, Hm; cutoff=1e-8,maxdim=16)
#         ψ = tdvp(Hm, ψ; vumps_kwargs...)
#     end
#     # 
    
    
# end
#
#  Environment tests
#
expected_e=[
    [0.25,-0.5/2,-0.25/3,-1.0/4,-0.75/5,-1.5/6,-1.25/7,-2/8],
    [0.0,-0.4279080101,0.0,-0.4279080101,0.0,-0.4279080101,0.0,-0.4279080101],
    [],
    [0.0,-0.4410581973,0.0,-0.4410581973,0.0,-0.4410581973,0.0,-0.4410581973]
    ]

eps=[1e-15,1e-9,0.0,2e-9]
let 
    println("----------------------------------------------")
    initstate(n) = isodd(n) ? "↑" : "↓"
    for N in  1:8
        for nex in [0,1,2]
            isodd(N) && nex>0 && continue
            s = siteinds("S=1/2", N; conserve_qns=false)
            si = infsiteinds(s)
            ψ = InfMPS(si, initstate)
            Hm = InfiniteMPOMatrix(Model("heisenberg"), si)
            ψ = tdvp(Hm, ψ; vumps_kwargs...)
            for _ in 1:nex
                ψ = subspace_expansion(ψ, Hm; cutoff=1e-8,maxdim=16)
                ψ = tdvp(Hm, ψ; vumps_kwargs...)
            end
            l = linkinds(only, ψ.AL)
            D=dim(l[1])
            println("Testing Ncell=$N, bond dimension D=$D")
            

            L,eₗ=left_environment(Hm,ψ) #Loic's version
            # @show abs(eₗ/N-expected_e[D][N]) nex D eps[D]
            @assert abs(eₗ/N-expected_e[D][N])<eps[D]
            R,eᵣ=right_environment(Hm,ψ) #Loic's version
            @assert abs(eᵣ/N-expected_e[D][N])<eps[D]


            H = InfiniteMPO(Model("heisenberg"), si)
            L,eₗ=left_environment1(H,ψ)
            # @show abs(eₗ/N-expected_e[D][N]) eₗ D eps[D]
            @assert abs(eₗ/N-expected_e[D][N])<eps[D]
            R,eᵣ=right_environment1(H,ψ)
            # @show abs(eᵣ/N-expected_eₗ[D][N]) D eps[D]
            @assert abs(eᵣ/N-expected_e[D][N])<eps[D]

            Hc=orthogonalize(H)
            L,eₗ=left_environment1(Hc.AL,ψ)
            @assert abs(eₗ/N-expected_e[D][N])<eps[D]
            L,eₗ=left_environment1(Hc.AR,ψ)
            @assert abs(eₗ/N-expected_e[D][N])<eps[D]
            R,eᵣ=right_environment1(Hc.AL,ψ)
            @assert abs(eᵣ/N-expected_e[D][N])<eps[D]
            R,eᵣ=right_environment1(Hc.AR,ψ)
            @assert abs(eᵣ/N-expected_e[D][N])<eps[D]


            Hc,BondSpectrums = truncate(H) 
            L,eₗ=left_environment1(Hc.AL,ψ)
            @assert abs(eₗ/N-expected_e[D][N])<eps[D]
            L,eₗ=left_environment1(Hc.AR,ψ)
            @assert abs(eₗ/N-expected_e[D][N])<eps[D]
            R,eᵣ=right_environment1(Hc.AL,ψ)
            @assert abs(eᵣ/N-expected_e[D][N])<eps[D]
            R,eᵣ=right_environment1(Hc.AR,ψ)
            @assert abs(eᵣ/N-expected_e[D][N])<eps[D]
        end
    end

    # 

    # ψ = InfiniteMPS(s;space=2)
    # for n in 1:N
    #     ψ[n] = randomITensor(inds(ψ[n]))
    # end
    # ψ = orthogonalize(ψ, :)
    # for n in 1:N
    #     @show norm(ψ.AL[n]*ψ.C[n] - ψ.C[n-1]*ψ.AR[n])
    # end
    #ψ = tdvp(Hm, ψ; vumps_kwargs...)
    # for _ in 1:1
    #     ψ = subspace_expansion(ψ, Hm; cutoff=1e-8,maxdim=16)
    #     ψ = tdvp(Hm, ψ; vumps_kwargs...)
    # end
    # pprint(Hc.AL[1])
    # L,R=environment(Hc.AL,ψ)    
    
    
    
end

nothing