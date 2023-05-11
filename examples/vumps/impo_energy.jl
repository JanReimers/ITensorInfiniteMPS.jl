using ITensors
using ITensorInfiniteMPS
using ITensorMPOCompression
using KrylovKit: linsolve
using Random 

Random.seed!(1234)

import ITensorInfiniteMPS: left_environment, right_environment, translatecell
import ITensorMPOCompression: slice, parse_links, parse_site

function translatecell(translator::Function, V::Vector{ITensor}, n::Integer)
    if n==0
        return V
    end
    V1=Vector{ITensor}()
    for T in V
        if length(inds(T))>0
            # @show n T translator
            Tt=translatecell(translatecelltags, T, n)
            # @show Tt
            push!(V1,Tt)
        else
            push!(V1,T)
        end
    end
    return V1
end

vumps_kwargs = (
      multisite_update_alg="sequential",
    #   multisite_update_alg="parallel",
      tol=1e-8,
      maxiter=50,
      outputlevel=1,
      time_step=-Inf,
    )


#initstate(n) = "↑"


# let
#     Eexact=(0.5 - 2 * log(2)) / 2
#     eᴸ, eᴿ=0.0,0.0
#     s = infsiteinds("S=1/2", N; initstate, conserve_szparity=true)
#     ψ = InfMPS(s, initstate)
#     for n in 1:N
#         ψ[n] = randomITensor(inds(ψ[n]))
#     end

#     # Form the Hamiltonian
#     H = InfiniteMPOMatrix(Model("heisenberg"), s)
#     ψ,_,_ = tdvp(H, ψ; vumps_kwargs...)
#     for _ in 1:4
#         ψ = subspace_expansion(ψ, H; cutoff=1e-8,maxdim=16)
#         ψ, eᴸ, eᴿ = tdvp(H, ψ; vumps_kwargs...)
#     end
#     @show Eexact eᴸ eᴿ
# end
function calculate_YLs(L::Vector{CelledVector{ITensor}},HL::InfiniteMPO,ψL::InfiniteMPS,b::Int64)
    Dw=size(L,1)
    N=size(L[Dw],1)
    ψL′ = dag(ψL)'
    YL=CelledVector{ITensor}(undef,N)
    for k in 1:N
        YL[k]=ITensor(0.0,inds(L[b+1][k]))
        # if b==1
        #     @show YL[k]
        # end
       
        il,ir=ITensorMPOCompression.parse_links(HL[k])
        for a′ in b+1:Dw
            Wapb=slice(HL[k],il=>a′,ir=>b)
            
            YL[k]+=L[a′][k-1]*ψL′[k]*Wapb*ψL[k]
            # if b==1
            #     @show Wapb a′ b il ir YL[k]
            # end
        end
    end
    return YL
end

function calculate_YRs(R::Vector{CelledVector{ITensor}},HR::InfiniteMPO,ψR::InfiniteMPS,a::Int64)
    Dw=size(R,1)
    N=size(R[Dw],1)
    ψR′ = dag(ψR)'
    YR=CelledVector{ITensor}(undef,N)
    for k in 1:N
        #@show a k inds(R[a-1][k]) inds(R[a-1][k-1]) inds(ψR[k])
        YR[k]=ITensor(0.0,inds(R[a-1][k-1]))
        # @show inds(YR[k])
        il,ir=ITensorMPOCompression.parse_links(HR[k])
        for b′ in 1:a-1
            Wabp=slice(HR[k],il=>a,ir=>b′)
            # @show inds(ψR′[k]) inds(ψR[k]) inds(R[b′][k]) 
            YR[k]+=ψR′[k]*Wabp*ψR[k]*R[b′][k]
            # @show inds(YR[k])
        end
    end
    return YR
end

function calculate_BL(YL::CelledVector{ITensor},ψL::InfiniteMPS,k::Int64)
    N=nsites(ψL)
    s = siteinds(only, ψL)
    δˢ(kk) = δ(dag(s[kk]), prime(s[kk]))
    ψL′ = dag(ψL)'
    B=YL[k-N+1]
    for k′ in k-N+2:k
        B=B*ψL′[k′]*δˢ(k′)*ψL[k′]+YL[k′]
    end
    return B
end
function calculate_BR(YR::CelledVector{ITensor},ψR::InfiniteMPS,k::Int64)
    N=nsites(ψR)
    s = siteinds(only, ψR)
    δˢ(kk) = δ(dag(s[kk]), prime(s[kk]))
    ψR′ = dag(ψR)'
    B=YR[k+N-1]
    for k′ in k+N-2:-1:k
        B=B*ψR′[k′]*δˢ(k′)*ψR[k′]+YR[k′]
    end
    return B
end

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
    𝕀=denseblocks(δˡ(k))
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
    # l′ = linkinds(only, ψ′.AL)
    r = linkinds(only, ψ.AR)
    # r′ = linkinds(only, ψ′.AR)
    s = siteinds(only, ψ)
    δʳ(kk) = δ(dag(r[kk]), prime(r[kk]))
    δˡ(kk) = δ(dag(l[kk]), prime(l[kk]))
    # δˡ(kk) = δ(l[kk], l′[kk])
    δˢ(kk) = δ(dag(s[kk]), prime(s[kk]))

    xT = translatecell(translator(ψ), x, 1) #xT = x shifted one unit cell to the right.
    TL=nothing
    for k′=k+N-1:-1:k
        xT*=ψ′.AR[k′]*δˢ(k′)*ψ.AR[k′]
        if isnothing(TL)
            TL=ψ′.AR[k′]*δˢ(k′)*ψ.AR[k′]
        else
            TL*=ψ′.AR[k′]*δˢ(k′)*ψ.AR[k′]
        end
    end
    L=ψ.C[k] * (ψ′.C[k]*δˡ(k))
    𝕀=denseblocks(δʳ(k))

    xL=𝕀*L*x # |𝕀)(L|x)
    #@show TL 𝕀*L

    return xT-xL
end

# function apply_TW(L::Vector{CelledVector{ITensor}},R::Vector{CelledVector{ITensor}},H::InfiniteMPO,ψ::InfiniteCanonicalMPS)
#     ψ′ =dag(ψ)'
#     l = linkinds(only, ψ.AL)
#     r = linkinds(only, ψ.AR)
#     s = siteinds(only, ψ)
#     δʳ(kk) = δ(dag(r[kk]), prime(r[kk]))
#     δˡ(kk) = δ(dag(l[kk]), prime(l[kk]))


#     Dw=size(L,1)
#     N=size(L[Dw],1)
#     L1=Vector{CelledVector{ITensor}}(undef,Dw)
#     R1=Vector{CelledVector{ITensor}}(undef,Dw)
#     for a in 1:Dw
#         L1[a]=CelledVector{ITensor}(undef,N)
#         R1[a]=CelledVector{ITensor}(undef,N)
#     end
#     for k in 1:N
#         il,ir=ITensorMPOCompression.parse_links(H[k])
#         for b in 1:Dw
#             Wbb=slice(H[k],il=>b,ir=>b)
         
#             L1[b][k]=L[b][k-1]*ψ′.AL[k]*Wbb*ψ.AL[k]
           
#             # b==1 && @show L1[b][k] 
#             for a in b+1:Dw
#                 Wab=slice(H[k],il=>a,ir=>b)
#                 L1[b][k]+=L[a][k-1]*ψ′.AL[k]*Wab*ψ.AL[k]
#                 # b==1 && @show a L1[b][k] 
#                 #@show a b Wab il ir L1[b][k] ψ′.AL[k]*Wab*ψ.AL[k]
#             end
#         end
#         for a in 1:Dw
#             Waa=slice(H[k],il=>a,ir=>a)
#             R1[a][k-1]=R[a][k]*ψ′.AR[k]*Waa*ψ.AR[k]
#             if scalar(Waa*Waa)==dim(s[k]) && a>1
#                 Lk=ψ.C[k] * (ψ′.C[k]*δˡ(k))
#                 𝕀k=denseblocks(δʳ(k))
#                 # @show a R1[a][k-1] R[a][k-1]*Lk*𝕀k
#                 R1[a][k-1]-=R[a][k-1]*Lk*𝕀k
#             end
#             for b in 1:a-1
#                 Wab=slice(H[k],il=>a,ir=>b)
#                 R1[a][k-1]+=R[b][k]*ψ′.AR[k]*Wab*ψ.AR[k]
#             end
#         end
#     end
#     for k in 1:N
#         il,ir=ITensorMPOCompression.parse_links(H[k])
#         for b in 1:Dw
#             Wbb=slice(H[k],il=>b,ir=>b)
#             if scalar(Wbb*Wbb)==dim(s[k]) && b<Dw
#                 # @show k L1[b][k] L[b][k]
#                 Rk=ψ.C[k] * (ψ′.C[k] * δʳ(k))
#                 𝕀k=denseblocks(δˡ(k))
#                 #@show Rk*𝕀k
#                 L1[b][k]-=L[b][k]*Rk*𝕀k
#                 # @show  L1[b][k]
#             end
#         end
#     end
#     return L1,R1
# end

function apply_TW_left(Lₖ₋₁::Vector{ITensor},Ŵ::ITensor,ψ::ITensor)
    ψ′ =dag(ψ)'
    il,ir=parse_links(Ŵ)
    dh,_,_=parse_site(Ŵ)
    @assert dim(il)==dim(ir)
    Dw=dim(il)
    Lₖ=Vector{ITensor}(undef,Dw)
    for b in Dw:-1:1
        for a in Dw:-1:b
            if isassigned(Lₖ₋₁,a) 
                Wab=slice(Ŵ,il=>a,ir=>b)
                # nab=norm(Wab)
                W2=scalar(Wab*dag(Wab))
                # if nab>=0.0
                    if !isassigned(Lₖ,b)
                        Lₖ[b]=emptyITensor()
                    end
                    Lₖ[b]+=Lₖ₋₁[a]*ψ′*Wab*ψ
                    @assert order(Lₖ[b])==2
                if W2==dh && a==b && a<Dw && a>1
                    @show dh a b Wab
                    @assert false
                end #if W2>0
            end # if L[a] assigned
        end # for a
    end # for b
    return Lₖ
end


function left_environment(H::InfiniteMPO, ψ::InfiniteCanonicalMPS; tol=1e-10)
    ψ′ =dag(ψ)'
    l = linkinds(only, ψ.AL)
    r = linkinds(only, ψ.AR)
    # s = siteinds(only, ψ)
    δʳ(kk) = δ(dag(r[kk]), prime(r[kk]))
    δˡ(kk) = δ(dag(l[kk]), prime(l[kk]))
    # dh=dim(s[1])
    il,ir=ITensorMPOCompression.parse_links(H[1])
    @assert dim(il)==dim(ir)
    Dw=dim(il)
    N=nsites(ψ)
    # Lₖ₋₁=Vector{ITensor}(undef,Dw)
    L₁=Vector{ITensor}(undef,Dw)
    
    #
    #  Solve for k=1
    #
    L₁[Dw]=δˡ(1) #Left eigen vector of TL (not TWL)
    for b1 in Dw-1:-1:1 #sweep backwards from Dw-1 down.
        #
        #  Load up all the know tensors from b1+1 to Dw.  Also translate one unit to the left.
        #
        Lₖ₋₁=Vector{ITensor}(undef,Dw)
        for b in Dw:-1:b1+1
            @assert isassigned(L₁,b) 
            Lₖ₋₁[b]=translatecell(translator(ψ), L₁[b], -1)
        end
        #
        #  Loop throught the unit cell and apply Tᵂₗ
        #
        for k in 2-N:1
            Lₖ₋₁=apply_TW_left(Lₖ₋₁,H[k],ψ.AL[k])
        end # for k
        @assert isassigned(Lₖ₋₁,b1) 
        L₁[b1]=Lₖ₋₁[b1] #save the new value.
    end #for b1

    # @show array.(L₁) inds.(L₁)
    localR = ψ.C[1] * δʳ(1) * ψ′.C[1] #to revise
    # @show localR
    eₗ = [0.0]
    eₗ[1] = (L₁[1] * localR)[]
    @show eₗ[1]
    L₁[1] += -(eₗ[1] * denseblocks(δˡ(1)))
    A = ALk(ψ, 1)
    L₁[1], info = linsolve(A, L₁[1], 1, -1; tol=tol)

    vN=[[emptyITensor() for n in 1:Dw] for n in 1:N]
    L=CelledVector{Vector{ITensor}}(vN,translatecell)
    L[1]=L₁

    #
    #  Now sweep throught the cell and evlaute all the L[k] form L[1]
    #
    for k in 2:N
        L[k]=apply_TW_left(L[k-1],H[k],ψ.AL[k])
    end
    #
    #  Verify that we get L[1] back from L[0]
    #
    L₁=apply_TW_left(L[0],H[1],ψ.AL[1])
    for b in 2:Dw #We know that L₁[1] is wrong
        @assert  norm(L₁[b]-L[1][b])==0.0
    end

    return L,eₗ[1]
end

function environment(H::InfiniteMPO, ψ::InfiniteCanonicalMPS; tol=1e-10)
    # ψ′ =dag(ψ)'
    l = linkinds(only, ψ.AL)
    # l′ = linkinds(only, ψ′.AL)
    r = linkinds(only, ψ.AR)
    # r′ = linkinds(only, ψ′.AR)
    s = siteinds(only, ψ)
    δʳ(kk) = δ(dag(r[kk]), prime(r[kk]))
    δˡ(kk) = δ(dag(l[kk]), prime(l[kk]))
    # δˡ(kk) = δ(l[kk], l′[kk])
    # δˢ(kk) = δ(dag(s[kk]), prime(s[kk]))

    il,ir=ITensorMPOCompression.parse_links(H[1])
    @assert dim(il)==dim(ir)
    Dw=dim(il)
    N=nsites(ψ)
    L=Vector{CelledVector{ITensor}}(undef,Dw)
    #L = [CelledVector{ITensor}(undef, N) for w in 1:Dw] #yields Vector{CelledVector{ITensor, typeof(translatecelltags)}}
    R=Vector{CelledVector{ITensor}}(undef,Dw)
    YL=Vector{CelledVector{ITensor}}(undef,Dw)
    YR=Vector{CelledVector{ITensor}}(undef,Dw)
    for a in 1:Dw
        L[a]=CelledVector{ITensor}(undef,N)
        R[a]=CelledVector{ITensor}(undef,N)
    end
    for k in 1:N
        L[Dw][k]=δˡ(k)
        R[1][k]=δʳ(k)
        #@show k inds(L[Dw][k]) inds(R[1][k])
    end


    for a in 2:Dw
        b=Dw-a+1
        YL[b]=calculate_YLs(L,H,ψ.AL,b)
        YR[a]=calculate_YRs(R,H,ψ.AR,a)
        for k in 1:N
            #@show k b inds(YL[b][k])
            #@show k a inds(YR[a][k])
            #
            #  Diagonal for L
            #
            il,ir=ITensorMPOCompression.parse_links(H[k])
            Wbb=slice(H[k],il=>b,ir=>b)
            nWbb=scalar(Wbb*Wbb)
            if nWbb==0
                L[b][k]=YL[b][k]
            elseif nWbb==dim(s[k])
                #@show YL[b]
                BLk=calculate_BL(YL[b],ψ.AL,k)
                #@show BLk
                #@show k b YL[b] BLk
                A = ALk(ψ, k)
                L[b][k], info = linsolve(A, BLk, 1, -1; tol=tol)
            else
                @show k b nWbb s[k] Wbb
                @assert false
            end
            #
            #  Diagonal for R
            #
            il,ir=ITensorMPOCompression.parse_links(H[k])
            Waa=slice(H[k],il=>a,ir=>a)
            nWaa=scalar(Waa*Waa)
            if nWaa==0
                R[a][k-1]=YR[a][k]
            elseif nWaa==dim(s[k]) #We hit a unit op on the diagonal.
                BRk=calculate_BR(YR[a],ψ.AR,k)
                # @show inds(BRk)
                A = ARk(ψ, k)
                R[a][k-1], info=linsolve(A, BRk, 1, -1; tol=tol)
            else
                @show k a nWaa s[k] Waa
                @assert false
            end
            # @show k-1 inds(R[a][k-1])
        end

    end
    for k=1:N
       el=scalar(L[1][k]*YL[1][k])
    #    @show inds(R[Dw][k]) inds(YR[Dw][k])
       er=scalar(R[Dw][k]*YR[Dw][k+1])
       @show el er
    end

    return L,R
end
expected_eₗ=[0.25,-0.5,-0.25,-1.0,-0.75,-1.5,-1.25,-2]

let 
    println("----------------------------------------------")
    initstate(n) = isodd(n) ? "↑" : "↓"
    for N in 1:8
        s = siteinds("S=1/2", N; conserve_qns=false)
        si = infsiteinds(s)
        ψ = InfMPS(si, initstate)
        H = InfiniteMPO(Model("heisenberg"), si)
        L,eₗ=left_environment(H,ψ)
        @assert abs(eₗ-expected_eₗ[N])<1e-15
        
        Hc=orthogonalize(H)
        L,eₗ=left_environment(Hc.AL,ψ)
        @assert abs(eₗ-expected_eₗ[N])<1e-15
        L,eₗ=left_environment(Hc.AR,ψ)
        @assert abs(eₗ-expected_eₗ[N])<1e-15

        Hc,BondSpectrums = truncate(H) 
        L,eₗ=left_environment(Hc.AL,ψ)
        @assert abs(eₗ-expected_eₗ[N])<1e-15
        L,eₗ=left_environment(Hc.AR,ψ)
        @assert abs(eₗ-expected_eₗ[N])<1e-15

        #@show array.(L)
    end

    # Hm = InfiniteMPOMatrix(Model("heisenberg"), si)
    # L,el=left_environment(Hm,ψ)
    # @show array.(L[1])  el

    # ψ = InfiniteMPS(s;space=2)
    # for n in 1:N
    #     ψ[n] = randomITensor(inds(ψ[n]))
    # end
    # ψ = orthogonalize(ψ, :)
    # for n in 1:N
    #     @show norm(ψ.AL[n]*ψ.C[n] - ψ.C[n-1]*ψ.AR[n])
    # end
    # @show inds(ψ.AL[1]) inds(ψ.AR[1])
    # R,er=right_environment(Hm,ψ)
    # @show R[1] R[2] er
    #ψ = tdvp(Hm, ψ; vumps_kwargs...)
    # for _ in 1:1
    #     ψ = subspace_expansion(ψ, Hm; cutoff=1e-8,maxdim=16)
    #     ψ = tdvp(Hm, ψ; vumps_kwargs...)
    # end
    # pprint(Hc.AL[1])
    # L,R=environment(Hc.AL,ψ)    
    
    
    # L1,R1=apply_TW(L,R,Hc.AR,ψ)
    # Dw=size(L,1)
    # for k in 1:2
    #     for a in 1:Dw
    #         if norm(L[a][k]-L1[a][k])>1e-12
    #             @show k a L[a][k] L1[a][k]
    #         end
    #         # if norm(R[a][k]-R1[a][k])>1e-12
    #         #     @show k a R[a][k] R1[a][k] 
    #         # end
    #     end
    # end   
end


nothing