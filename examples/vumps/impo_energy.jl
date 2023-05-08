using ITensors
using ITensorInfiniteMPS
using ITensorMPOCompression
using KrylovKit: linsolve
using Random 

Random.seed!(1234)

import ITensorMPOCompression: slice

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
       
        il,ir=ITensorMPOCompression.parse_links(HL[k])
        for a′ in b+1:Dw
            Wapb=slice(HL[k],il=>a′,ir=>b)
            
            YL[k]+=L[a′][k-1]*ψL′[k]*Wapb*ψL[k]
            
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
    l′ = linkinds(only, ψ′.AL)
    r = linkinds(only, ψ.AR)
    r′ = linkinds(only, ψ′.AR)
    s = siteinds(only, ψ)
    δʳ(kk) = δ(dag(r[kk]), prime(r[kk]))
    δˡ(kk) = δ(l[kk], l′[kk])
    δˢ(kk) = δ(dag(s[kk]), prime(s[kk]))
    xT = translatecell(translator(ψ), x, -1) #xT = x shifted one unit cell to the left.
    for k′=k-N+1:k
        xT*=ψ′.AL[k′]*δˢ(k′)*ψ.AL[k′]
    end
    R=ψ.C[k] * (ψ′.C[k] * δʳ(k))
    𝕀=denseblocks(δˡ(k))
    xR=x*R*𝕀
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
    l′ = linkinds(only, ψ′.AL)
    r = linkinds(only, ψ.AR)
    r′ = linkinds(only, ψ′.AR)
    s = siteinds(only, ψ)
    δʳ(kk) = δ(dag(r[kk]), prime(r[kk]))
    δˡ(kk) = δ(l[kk], l′[kk])
    δˢ(kk) = δ(dag(s[kk]), prime(s[kk]))

    xT = translatecell(translator(ψ), x, 1) #xT = x shifted one unit cell to the right.
    for k′=k+N-1:-1:k
        xT*=ψ′.AR[k′]*δˢ(k′)*ψ.AR[k′]
    end
    L=ψ.C[k] * (ψ′.C[k]*δˡ(k))
    𝕀=denseblocks(δʳ(k))

    xL=𝕀*L*x # |𝕀)(L|x)

    return xT-xL
end

function environment(H::InfiniteCanonicalMPO, ψ::InfiniteCanonicalMPS; tol=1e-10)
    ψ′ =dag(ψ)'
    l = linkinds(only, ψ.AL)
    l′ = linkinds(only, ψ′.AL)
    r = linkinds(only, ψ.AR)
    r′ = linkinds(only, ψ′.AR)
    s = siteinds(only, ψ)
    δʳ(kk) = δ(dag(r[kk]), prime(r[kk]))
    δˡ(kk) = δ(l[kk], l′[kk])
    δˢ(kk) = δ(dag(s[kk]), prime(s[kk]))

    il,ir=ITensorMPOCompression.parse_links(H.AL[1])
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
        @show k inds(L[Dw][k]) inds(R[1][k])
    end
    for a in 2:Dw
        b=Dw-a+1
        YL[b]=calculate_YLs(L,H.AL,ψ.AL,b)
        YR[a]=calculate_YRs(R,H.AR,ψ.AR,a)
        for k in 1:N
            #@show k b inds(YL[b][k])
            #@show k a inds(YR[a][k])

            il,ir=ITensorMPOCompression.parse_links(H.AL[k])
            Wbb=slice(H.AL[k],il=>b,ir=>b)
            nWbb=scalar(Wbb*Wbb)
            if nWbb==0
                L[b][k]=YL[b][k]
            elseif nWbb==dim(s[k])
                BLk=calculate_BL(YL[b],ψ.AL,k)
                A = ALk(ψ, k)
                L[b][k], info = linsolve(A, BLk, 1, -1; tol=tol)
            else
                @show k b nWbb s[k] Wbb
                @assert false
            end
            il,ir=ITensorMPOCompression.parse_links(H.AR[k])
            Waa=slice(H.AR[k],il=>a,ir=>a)
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

let 
    initstate(n) = isodd(n) ? "↑" : "↓"
    N=2
    s = siteinds("S=1/2", N; conserve_qns=false)
    si = infsiteinds(s)
    ψ = InfMPS(si, initstate)
    # ψ = InfiniteMPS(s;space=2)
    # for n in 1:N
    #     ψ[n] = randomITensor(inds(ψ[n]))
    # end
    # ψ = orthogonalize(ψ, :)
    # for n in 1:N
    #     @show norm(ψ.AL[n]*ψ.C[n] - ψ.C[n-1]*ψ.AR[n])
    # end
    @show inds(ψ.AL[1]) inds(ψ.AR[1])

    H = InfiniteMPO(Model("heisenberg"), si)
    Hc,BondSpectrums = truncate(H) #Use default cutoff,C is now diagonal
    Hm = InfiniteMPOMatrix(Model("heisenberg"), si)
        ψ = tdvp(Hm, ψ; vumps_kwargs...)
        for _ in 1:4
            ψ = subspace_expansion(ψ, Hm; cutoff=1e-8,maxdim=16)
            ψ = tdvp(Hm, ψ; vumps_kwargs...)
        end
    L,R=environment(Hc,ψ)    
   
   
end


nothing