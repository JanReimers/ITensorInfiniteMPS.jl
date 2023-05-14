using ITensors
using ITensorInfiniteMPS
using ITensorMPOCompression
using KrylovKit: linsolve
using Random 

Random.seed!(1234)

using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%1.6f", f) #dumb way to control float output

import ITensorInfiniteMPS: left_environment, right_environment, vumps_solver,tdvp_iteration_sequential
import ITensorInfiniteMPS: ALk, ARk
# , translatecell, vumps_solver,     tdvp_iteration_sequential,H⁰,H¹,ortho_polar
import ITensorMPOCompression: assign!, slice


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

let 
    println("----------------------------------------------")
    # initstate(n) = isodd(n) ? "↑" : "↓"
    initstate(n) = "↑"
    N,nex=1,2
    # s = siteinds("S=1/2", N; conserve_qns=false)
    si = infsiteinds("S=1/2", N;initstate,conserve_szparity=true)
    ψ = InfMPS(si, initstate)
    # Hm = InfiniteMPOMatrix(Model("heisenberg"), si)
    # ψ = tdvp(Hm, ψ; vumps_kwargs...)
    # for _ in 1:nex
    #     ψ = subspace_expansion(ψ, Hm; cutoff=1e-8,maxdim=16)
    #     ψ = tdvp(Hm, ψ; vumps_kwargs...)
    # end
    # ψ,(el,er)=ITensorInfiniteMPS.tdvp_iteration_sequential(vumps_solver,Hm,ψ;time_step=-Inf)
    # @show el er 

    # H = InfiniteMPOMatrix(Model("heisenberg"), si)
    H = InfiniteMPO(Model("ising"), si;J=1.0, h=1.2)
    # ψ,(el,er)=tdvp_iteration_sequential(vumps_solver,H,ψ;time_step=-Inf)
    # @show el er 
    ψ = tdvp(H, ψ; vumps_kwargs...)
    for i in 1:nex
        println("\n--------------Subspace expansion---------------\n")
        ψ = subspace_expansion(ψ, H; cutoff=1e-8,maxdim=16)
        println("\n--------------tdvp---------------\n")
        ψ = tdvp(H, ψ; vumps_kwargs...)
    end
    # 
    
    
end
#
#  Environment tests
#
function check1(Lm::CelledVector,L::CelledVector,lₕ::CelledVector,offset::Int64=0)
    N=length(Lm)
    
    for k in 1:N
        Lmv=Vector{ITensor}()
        is=inds(Lm[k+offset][1])
        for Lmi in Lm[k+offset]
            if order(Lmi)==2
                push!(Lmv,Lmi)
            elseif order(Lmi)==3
                il=noncommonind(Lmi,is)
                for i in 1:dim(il)
                    push!(Lmv,slice(Lmi,il=>i))
                end
            else
                @assert false
            end
        end
        il=lₕ[k]
        for i in 1:dim(il)
            Lki=slice(L[k],il=>i)
            if norm(Lki-Lmv[i])>1e-14
                @show k i Lki Lmv[i]
            end
        end
    end
    # @assert length(L)==Lm
    # @show typeof(Lm)
end

expected_e=[
    [0.25,-0.5/2,-0.25/3,-1.0/4,-0.75/5,-1.5/6,-1.25/7,-2/8],
    [0.0,-0.4279080101,0.0,-0.4279080101,0.0,-0.4279080101,0.0,-0.4279080101],
    [],
    [0.0,-0.4410581973,0.0,-0.4410581973,0.0,-0.4410581973,0.0,-0.4410581973]
    ]

eps=[1e-15,1e-9,0.0,2e-9]

# let 
#     println("----------------------------------------------")
#     initstate(n) = isodd(n) ? "↑" : "↓"
#     for N in  2:2
#         for nex in [1]
#             isodd(N) && nex>0 && continue
#             s = siteinds("S=1/2", N; conserve_qns=false)
#             si = infsiteinds(s)
#             ψ = InfMPS(si, initstate)
#             Hm = InfiniteMPOMatrix(Model("heisenberg"), si)
#             ψ = tdvp(Hm, ψ; vumps_kwargs...)
#             for _ in 1:nex
#                 ψ = subspace_expansion(ψ, Hm; cutoff=1e-8,maxdim=16)
#                 ψ = tdvp(Hm, ψ; vumps_kwargs...)
#             end
#             l = linkinds(only, ψ.AL)
#             D=dim(l[1])
#             println("Testing Ncell=$N, bond dimension D=$D")
            

#             Lm,eₗ=left_environment(Hm,ψ) #Loic's version
#             # @show abs(eₗ/N-expected_e[D][N]) nex D eps[D]
#             @assert abs(eₗ/N-expected_e[D][N])<eps[D]
#             Rm,eᵣ=right_environment(Hm,ψ) #Loic's version
#             @assert abs(eᵣ/N-expected_e[D][N])<eps[D]


#             H = InfiniteMPO(Model("heisenberg"), si)
#             L,eₗ=left_environment(H,ψ)
#             check1(Lm,L,linkinds(only, H))
#             # @show abs(eₗ/N-expected_e[D][N]) eₗ D eps[D]
#             @assert abs(eₗ/N-expected_e[D][N])<eps[D]
#             R,eᵣ=right_environment(H,ψ)
#             # @show abs(eᵣ/N-expected_eₗ[D][N]) D eps[D]
#             @assert abs(eᵣ/N-expected_e[D][N])<eps[D]
#             check1(Rm,R,linkinds(only, H),1)

#             # Hc=orthogonalize(H)
#             # L,eₗ=left_environment(Hc.AL,ψ)
#             # @assert abs(eₗ/N-expected_e[D][N])<eps[D]
#             # L,eₗ=left_environment(Hc.AR,ψ)
#             # @assert abs(eₗ/N-expected_e[D][N])<eps[D]
#             # R,eᵣ=right_environment(Hc.AL,ψ)
#             # @assert abs(eᵣ/N-expected_e[D][N])<eps[D]
#             # R,eᵣ=right_environment(Hc.AR,ψ)
#             # @assert abs(eᵣ/N-expected_e[D][N])<eps[D]


#             # Hc,BondSpectrums = truncate(H) 
#             # L,eₗ=left_environment(Hc.AL,ψ)
#             # @assert abs(eₗ/N-expected_e[D][N])<eps[D]
#             # L,eₗ=left_environment(Hc.AR,ψ)
#             # @assert abs(eₗ/N-expected_e[D][N])<eps[D]
#             # R,eᵣ=right_environment(Hc.AL,ψ)
#             # @assert abs(eᵣ/N-expected_e[D][N])<eps[D]
#             # R,eᵣ=right_environment(Hc.AR,ψ)
#             # @assert abs(eᵣ/N-expected_e[D][N])<eps[D]
#         end
#     end

#     # 

#     # ψ = InfiniteMPS(s;space=2)
#     # for n in 1:N
#     #     ψ[n] = randomITensor(inds(ψ[n]))
#     # end
#     # ψ = orthogonalize(ψ, :)
#     # for n in 1:N
#     #     @show norm(ψ.AL[n]*ψ.C[n] - ψ.C[n-1]*ψ.AR[n])
#     # end
#     #ψ = tdvp(Hm, ψ; vumps_kwargs...)
#     # for _ in 1:1
#     #     ψ = subspace_expansion(ψ, Hm; cutoff=1e-8,maxdim=16)
#     #     ψ = tdvp(Hm, ψ; vumps_kwargs...)
#     # end
#     # pprint(Hc.AL[1])
#     # L,R=environment(Hc.AL,ψ)    
    
    
    
# end

nothing