
import Base: inv

struct InfiniteCanonicalMPO <: AbstractInfiniteMPS
    H0::InfiniteMPO #uncompressed version used for triangular environment algo.
    AL::InfiniteMPO
    GLR::CelledVector{ITensor} #Gauge transform betwee AL and AR
    AR::InfiniteMPO
    G::CelledVector{ITensor} #Gauge transform betwee H0 and AL
end

function InfiniteCanonicalMPO(H0::InfiniteMPO,HL::reg_form_iMPO,GLR::CelledVector{ITensor},HR::reg_form_iMPO,G::CelledVector{ITensor})
    return InfiniteCanonicalMPO(H0,InfiniteMPO(HL),GLR,InfiniteMPO(HR),G)
end

Base.length(H::InfiniteCanonicalMPO)=length(H.GLR)
ITensors.data(H::InfiniteCanonicalMPO)=H.AL
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
function check_gauge(H::InfiniteCanonicalMPO)::Float64
    eps2=0.0
    for k in eachindex(H)
        eps2+=norm(H.GLR[k - 1] * H.AR[k] - H.AL[k] * H.GLR[k])^2
    end
    return sqrt(eps2)
end

#
#  Check the gauge transform between AR and H0 (the raw unmodified Hamiltonian)
#
function check_gauge(H::InfiniteCanonicalMPO,H0::InfiniteMPO)::Float64
    eps2=0.0
    for k in eachindex(H)
        eps2+=norm(H.G[k - 1] * H.AR[k] - H0[k] * H.G[k])^2
    end
    # @show norm(Ho.G[0]*Ho.AR[1]-H[1]*Ho.G[1])
    return sqrt(eps2)
end

function ITensors.orthogonalize(H0::InfiniteMPO;kwargs...)::InfiniteCanonicalMPO
    HL,GLR,HR,G=orthogonalize(reg_form_iMPO(H0);kwargs...)
    return InfiniteCanonicalMPO(H0,HL,GLR,HR,G)
end
function ITensors.orthogonalize(Hi::reg_form_iMPO;kwargs...)
    HL=copy(Hi) #not HL yet, but will be after two ortho calls.
    GL=orthogonalize!(HL, left; kwargs...)
    HR = copy(HL)
    GR = orthogonalize!(HR,right; kwargs...)
    return HL,GR,HR,full_ortho_gauge(GL,GR)
end

function ITensors.truncate(H0::InfiniteMPO;kwargs...)::Tuple{InfiniteCanonicalMPO,bond_spectrums}
    HL, HR, Ss, Gfull, ss = truncate!(reg_form_iMPO(H0);kwargs...)
    return InfiniteCanonicalMPO(H0,HL,Ss,HR,Gfull),ss
end

#
#  Assumes we first did right orth H0-->HR and the a left orth HR-->HL
#  The H's should satisfy: H0[K]*G[K]-G[k-1]*HL[k]
#
function full_ortho_gauge(GL::CelledVector{ITensor},GR::CelledVector{ITensor})
    @assert length(GL)==length(GR)
    N=length(GL)
    G=CelledVector{ITensor}(undef,N)
    for k in 1:N
        G[k]=Base.inv(GL[k])*GR[k]
        @assert order(G[k])==2
    end
    return G
end


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


