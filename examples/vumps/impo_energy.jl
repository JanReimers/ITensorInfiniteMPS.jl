using ITensors
using ITensorInfiniteMPS
using ITensorMPOCompression

import ITensorMPOCompression: slice

vumps_kwargs = (
      multisite_update_alg="sequential",
    #   multisite_update_alg="parallel",
      tol=1e-8,
      maxiter=50,
      outputlevel=0,
      time_step=-Inf,
    )


#initstate(n) = "↑"
initstate(n) = isodd(n) ? "↑" : "↓"
nsites=2

# let
#     Eexact=(0.5 - 2 * log(2)) / 2
#     eᴸ, eᴿ=0.0,0.0
#     s = infsiteinds("S=1/2", nsites; initstate, conserve_szparity=true)
#     ψ = InfMPS(s, initstate)
#     for n in 1:nsites
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

let 
    s = siteinds("S=1/2", nsites; conserve_qns=false)
    si = infsiteinds(s)
    # ψ = InfMPS(si, initstate)
    ψ = InfiniteMPS(s;space=2)
    for n in 1:nsites
        ψ[n] = randomITensor(inds(ψ[n]))
    end
    ψ = orthogonalize(ψ, :)
    for n in 1:nsites
        @show norm(ψ.AL[n]*ψ.C[n] - ψ.C[n-1]*ψ.AR[n])
    end
    ψ′ = dag(ψ)'

    l = linkinds(only, ψ.AL)
    l′ = linkinds(only, ψ′.AL)
    r = linkinds(only, ψ.AR)
    r′ = linkinds(only, ψ′.AR)
    s = siteinds(only, ψ)
    δʳ(n) = δ(dag(r[n]), prime(r[n]))
    δˡ(n) = δ(l[n], l′[n])
    δˢ(n) = δ(dag(s[n]), prime(s[n]))
    
    @show inds(ψ.AL[1]) inds(ψ′.AL[1]) l l′ r r′ s inds(δˡ(1)) inds(δʳ(0)) inds(δʳ(1))

    H = InfiniteMPO(Model("heisenberg"), si)
   
    # TWL_1=H[1]*dag(prime(ψ.AL[1],si[1]))*ψ.AL[1]
    # @show inds(TWL_1)
    # il,ir=inds(TWL_1)
    # pprint(il,TWL_1,ir)

    Hc,BondSpectrums = truncate(H) #Use default cutoff,C is now diagonal
    il,ir=ITensorMPOCompression.parse_links(Hc.AL[1])
    #@show il,ir
    @assert dim(il)==dim(ir)
    Dw=dim(il)
    N=nsites
    L=Vector{CelledVector{ITensor}}(undef,Dw)
    R=Vector{CelledVector{ITensor}}(undef,Dw)
    YL,YR=Matrix{ITensor}(undef,N,Dw),Matrix{ITensor}(undef,N,Dw)
    for a in 1:Dw
        L[a]=CelledVector{ITensor}(undef,N)
        R[a]=CelledVector{ITensor}(undef,N)
    end
    for k in 1:N
        L[Dw][k]=δˡ(k)
        R[1][k]=δʳ(k)
    end

    
    for a in 2:Dw
        b=Dw-a+1
        for k in 1:N
            @show k b 
            YL[k,b]=ITensor(0.0,inds(L[b+1][k]))
            il,ir=ITensorMPOCompression.parse_links(Hc.AL[k])
            for a′ in b+1:Dw
                Wapb=slice(Hc.AL[k],il=>a′,ir=>b)
                YL[k,b]+=L[a′][k-1]*ψ′.AL[k]*Wapb*ψ.AL[k]
            end
            YR[k,a]=ITensor(0.0,inds(R[a-1][k-1]))
            @show k a inds(YR[k,a])
            il,ir=ITensorMPOCompression.parse_links(Hc.AR[k])
            for b′ in 1:a-1
                Wabp=slice(Hc.AR[k],il=>a,ir=>b′)
                t=ψ′.AR[k]*Wabp*ψ.AR[k]
                @show  b′ inds(t) inds(R[b′][k])
                YR[k,a]+=ψ′.AR[k]*Wabp*ψ.AR[k]*R[b′][k]
            end
        end
        for k in 1:N
            il,ir=ITensorMPOCompression.parse_links(Hc.AL[k])
            Wbb=slice(Hc.AL[k],il=>b,ir=>b)
            if norm(Wbb)==0
                L[b][k]=YL[k,b]
            else
                @show k b Wbb
                @assert false
            end
            il,ir=ITensorMPOCompression.parse_links(Hc.AR[k])
            Waa=slice(Hc.AR[k],il=>a,ir=>a)
            if norm(Waa)==0
                R[a][k-1]=YR[k,a]
            else
                @assert false
            end
        end

    end

    a,b=Dwr-1,Dwl
    Wba=ITensorMPOCompression.slice(Hc.AL[1],il=>b,ir=>a)
    YL[a]=L[b]*ψ′.AL[1]*Wba*ψ.AL[1]
    @show inds(YL[Dwr-1])
    b=Dwl-1
    Wba=ITensorMPOCompression.slice(Hc.AL[1],il=>b,ir=>a)
    @assert norm(Wba)==0.0
    L[a]=YL[a]
    @show inds(L[Dwl]) inds(L[Dwl-1])
    a,b=Dwr-2,Dwl-1
    Wba=ITensorMPOCompression.slice(Hc.AL[1],il=>b,ir=>a)
    YL[a]= L[b]*ψ′.AL[1]*Wba*ψ.AL[1]
    b=Dwl
    Wba=ITensorMPOCompression.slice(Hc.AL[1],il=>b,ir=>a)
    # YL[a]+=L[b]*ψ′.AL[1]*Wba*ψ.AL[1]
    # @show YL[a]

    # ilψ,irψ=ITensorMPOCompression.parse_links(ψ.AL[1])
    # Cl=combiner(il,ilψ,ilψ')
    # Cr=combiner(ir,irψ,irψ')
    # pprint(Hc.AL[1])
    # TWL_1=Hc.AL[1]*dag(ψ.AL[1]')*ψ.AL[1]*Cl*Cr


    # @show inds(TWL_1)
    # il,ir=inds(TWL_1)
    # pprint(il,TWL_1,ir)
    # d,U=eigen(TWL_1,combinedind(Cl),combinedind(Cr))
    # @show diag(d)
end
nothing