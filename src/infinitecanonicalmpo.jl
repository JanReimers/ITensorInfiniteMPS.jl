
import Base: inv

struct InfiniteCanonicalMPO <: AbstractInfiniteMPS
    H0::InfiniteMPO #uncompressed version used for triangular environment algo.
    AL::InfiniteMPO
    AR::InfiniteMPO
    GLR::CelledVector{ITensor} #Gauge transform betwee AL and AR
    G0R::CelledVector{ITensor} #Gauge transform betwee H0 and AR. Used to get L/R environments
end

function InfiniteCanonicalMPO(H0::InfiniteMPO,HL::reg_form_iMPO,HR::reg_form_iMPO,GLR::CelledVector{ITensor},G0R::CelledVector{ITensor})
    @assert translator(H0)==translator(HL)
    @assert translator(H0)==translator(HR)
    @assert translator(H0)==translator(GLR)
    @assert translator(H0)==translator(G0R)
    return InfiniteCanonicalMPO(H0,InfiniteMPO(HL),InfiniteMPO(HR),GLR,G0R)
end

#
#  Vumps can work with either H.AL or H.AR we just need to pick one in the data/getindex overloads.  
#
Base.length(H::InfiniteCanonicalMPO)=length(H.GLR)
ITensors.data(H::InfiniteCanonicalMPO)=H.AR
ITensorInfiniteMPS.isreversed(::InfiniteCanonicalMPO)=false
Base.getindex(H::InfiniteCanonicalMPO, n::Int64)=getindex(H.AR,n)

function ITensorMPOCompression.get_Dw(H::InfiniteCanonicalMPO)
    return get_Dw(H.AR)
end

function check_ortho(H::InfiniteCanonicalMPO)::Bool
    return check_ortho(H.AL,left) && check_ortho(H.AR,right)
end

#
#  Check the gauge transform between AR and AL
#
function check_gauge_LR(H::InfiniteCanonicalMPO)::Float64
    eps2=0.0
    for k in eachindex(H)
        eps2+=norm(H.GLR[k - 1] * H.AR[k] - H.AL[k] * H.GLR[k])^2
    end
    return sqrt(eps2)
end

#
#  Check the gauge transform between AR and H0 (the raw unmodified Hamiltonian)
#
function check_gauge_0R(H::InfiniteCanonicalMPO,H0::InfiniteMPO)::Float64
    eps2=0.0
    for k in eachindex(H)
        eps2+=norm(H.G0R[k - 1] * H.AR[k] - H0[k] * H.G0R[k])^2
    end
    # @show norm(Ho.G[0]*Ho.AR[1]-H[1]*Ho.G[1])
    return sqrt(eps2)
end

function ITensors.orthogonalize(H0::InfiniteMPO;kwargs...)::InfiniteCanonicalMPO
    HL,HR,GLR,G0R=orthogonalize(reg_form_iMPO(H0);kwargs...)
    return InfiniteCanonicalMPO(H0,HL,HR,GLR,G0R)
end
function ITensors.orthogonalize(Hi::reg_form_iMPO;kwargs...)
    HL=copy(Hi) #not HL yet, but will be after two ortho calls.
    @assert translator(Hi)==translator(HL)
    G0L=orthogonalize!(HL, left; kwargs...)
    @assert translator(Hi)==translator(HL)
    @assert translator(Hi)==translator(G0L)
    HR = copy(HL)
    @assert translator(Hi)==translator(HR)
    GLR = orthogonalize!(HR,right; kwargs...)
    @assert translator(Hi)==translator(HR)
    @assert translator(Hi)==translator(GLR)
    G0LR=full_ortho_gauge(G0L,GLR)
    @assert translator(Hi)==translator(G0LR)
    #
    #  At this point HL is likely to have a larger bond dimension than HR because the second sweep matters for reducing Dw.
    #  Truncation will make this discrepency moot.
    #  If we need a reduced HL for some reason then one more sweep as below can be uncommented.
    # HL=copy(HR)
    # GRL = orthogonalize!(HL,left; kwargs...)
    return HL,HR,GLR,G0LR
end

function ITensors.truncate(H0::InfiniteMPO;kwargs...)::Tuple{InfiniteCanonicalMPO,bond_spectrums}
    HL, HR, Ss, Gfull, ss = truncate!(reg_form_iMPO(H0);kwargs...)
    return InfiniteCanonicalMPO(H0,HL,HR,Ss,Gfull),ss
end

#
#  Assumes we first did right orth H0-->HR and the a left orth HR-->HL
#  The H's should satisfy: H0[K]*G[K]-G[k-1]*HL[k]
#
function full_ortho_gauge(G0L::CelledVector{ITensor},GLR::CelledVector{ITensor})
    @assert length(G0L)==length(GLR)
    @assert translator(G0L)==translator(GLR)
    N=length(G0L)
    G0R=CelledVector{ITensor}(undef,N,translator(G0L))
    for k in 1:N
        G0R[k]=Base.inv(G0L[k])*GLR[k]
        @assert order(G0R[k])==2
    end
    return G0R
end

#
#  Penrose inverse
#
function Base.inv(A::ITensor;tol=1e-12,kwargs...)::ITensor
@assert order(A)==2

U,s,V=svd(A,ind(A,1);kwargs...)
if minimum(diag(s))<tol
    @warn("Trying to solve near singular system. diag(s)=$(diag(s))")
end
return dag(V)*invdiag(s)*dag(U)
end

function invdiag(s::ITensor)
    return itensor(invdiag(tensor(s)))
end

function invdiag(s::DiagTensor)
    # creating a DiagTensor directly seems to be very diffficult
    sinv=tensor(diagITensor(dag(inds(s))))
    for i in 1:diaglength(s)
        s1=1.0/NDTensors.getdiagindex(s,i)
        NDTensors.setdiagindex!(sinv,s1,i)
    end
    return sinv
end

function invdiag(s::DiagBlockSparseTensor)
    sinv=DiagBlockSparseTensor(nzblocks(s),dag(inds(s)))
    for i in 1:diaglength(s)
        s1=1.0/NDTensors.getdiagindex(s,i)
        NDTensors.setdiagindex!(sinv,s1,i)
    end
    return sinv
end


