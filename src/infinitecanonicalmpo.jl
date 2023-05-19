
struct InfiniteCanonicalMPO <: AbstractInfiniteMPS
    AL::InfiniteMPO
    GLR::CelledVector{ITensor} #Gauge transform betwee AL and AR
    AR::InfiniteMPO
    G::CelledVector{ITensor} #Gauge transform betwee H0 and AL
end

function InfiniteCanonicalMPO(HL::reg_form_iMPO,GLR::CelledVector{ITensor},HR::reg_form_iMPO,G::CelledVector{ITensor})
    return InfiniteCanonicalMPO(InfiniteMPO(HL),GLR,InfiniteMPO(HR),G)
end

Base.length(H::InfiniteCanonicalMPO)=length(H.C)
ITensors.data(H::InfiniteCanonicalMPO)=H.AL
ITensorInfiniteMPS.isreversed(::InfiniteCanonicalMPO)=false
Base.getindex(H::InfiniteCanonicalMPO, n::Int64)=getindex(H.AL,n)

function ITensorMPOCompression.check_ortho(H::InfiniteCanonicalMPO)::Bool
    return check_ortho(H.AL,left) && check_ortho(H.AR,right)
end

function check_gauge(H::InfiniteCanonicalMPO)::Float64
    eps2=0.0
    for n in eachindex(H)
        eps2+=norm(H.GLR[n - 1] * H.AR[n] - H.AL[n] * H.GLR[n])^2
    end
    return sqrt(eps2)
end

function ITensors.orthogonalize(Hi::InfiniteMPO;kwargs...)::InfiniteCanonicalMPO
    HL=reg_form_iMPO(Hi) #not HL yet, but will be after two ortho calls.
    # H0=copy(HL)
    GL=orthogonalize!(HL, left; kwargs...)
    HR = copy(HL)
    GR = orthogonalize!(HR,right; kwargs...)
    return InfiniteCanonicalMPO(HL,GR,HR,full_ortho_gauge(GL,GR))
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
        GL_inv=ITensor(transpose(pinv(array(GL[k]))),inds(GL[k])) #Penrose inverse for rectangular.
        G[k]=GL_inv*GR[k]
        @assert order(G[k])==2
    end
    return G
end

function ITensors.truncate(Hi::InfiniteMPO;kwargs...)::Tuple{InfiniteCanonicalMPO,bond_spectrums}
    HL, HR, Ss, ss = truncate!(reg_form_iMPO(Hi);kwargs...)
    return InfiniteCanonicalMPO(HL,Ss,HR,Ss),ss
end