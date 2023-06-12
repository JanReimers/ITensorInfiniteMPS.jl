

@doc """
    struct InfiniteCanonicalMPO <: AbstractInfiniteMPS <: ITensors.AbstractMPS 

# Fields
- H0 ::`InfiniteMPO`  Original iMPO.  Should lower triangular and lower regular form.
- AL ::`InfiniteMPO`  Left  orthogonal/canonical or truncated form of H0
- AR ::`InfiniteMPO`  Right orthogonal/canonical or truncated form of H0
- GLR::`CelledVector{ITensor}`  Gauge transform from AL --> AR
- G0R::`CelledVector{ITensor}`  Gauge transform from H0 --> AR
    
# Description
    This struct stores all required results of an orthogonalization or truncation operation.
        The VUMPS code requires a lower triangular `InfiniteMPO` in order to calculate L/R environments.
    For a number of reasons, orthogonalization and trunction will in general destroy the triangular structure.
    One way to get the L/R environments from a truncated (non-triangular) `InfiniteMPO` is using an eigen solver.
    Another less brute force approach is to calculate L&R from the un-compressed `InfiniteMPO`, H0, and then use the gauge
    Transform between H0 and AR to convert L/R into enviroments for the truncated AR
    
"""
struct InfiniteCanonicalMPO <: AbstractInfiniteMPS
    H0::InfiniteMPO #uncompressed version used for triangular environment algo.
    AL::InfiniteMPO #Left canonical form
    AR::InfiniteMPO #Right canonical form
    GLR::CelledVector{ITensor} #Gauge transform betwee AL and AR
    G0R::CelledVector{ITensor} #Gauge transform betwee H0 and AR. Used to get L/R environments for AR
end

#
# For systems like fqhe with non-trivial translators, it is very easy to lose the translator end up with the default translator.
# We check everything is consistent at construction time.
#
function InfiniteCanonicalMPO(H0::InfiniteMPO,HL::reg_form_iMPO,HR::reg_form_iMPO,GLR::CelledVector{ITensor},G0R::CelledVector{ITensor})
    @assert translator(H0)==translator(HL)
    @assert translator(H0)==translator(HR)
    @assert translator(H0)==translator(GLR)
    @assert translator(H0)==translator(G0R)
    return InfiniteCanonicalMPO(H0,InfiniteMPO(HL),InfiniteMPO(HR),GLR,G0R)
end

#--------------------------------------------------------------------------------------------
#
# Generic overlads
#
Base.length(H::InfiniteCanonicalMPO)=length(H.GLR)
ITensors.data(H::InfiniteCanonicalMPO)=H.AR
ITensorInfiniteMPS.isreversed(::InfiniteCanonicalMPO)=false
#  Vumps can work with either H.AL or H.AR we just need to pick one in the data/getindex overloads.  
Base.getindex(H::InfiniteCanonicalMPO, n::Int64)=getindex(H.AR,n)

function ITensorMPOCompression.get_Dw(H::InfiniteCanonicalMPO)
    return get_Dw(H.AR)
end

#-----------------------------------------------------------------------
#
#  Check ortho and various gauge relations
#
function check_ortho(H::InfiniteCanonicalMPO)::Bool
    return check_ortho(H.AL,left) && check_ortho(H.AR,right)
end

#  Check the gauge transform between AR and AL
function check_gauge_LR(H::InfiniteCanonicalMPO)::Float64
    eps2=0.0
    for k in eachindex(H)
        eps2+=norm(H.GLR[k - 1] * H.AR[k] - H.AL[k] * H.GLR[k])^2
    end
    return sqrt(eps2)
end

#  Check the gauge transform between AR and H0 (the raw unmodified Hamiltonian)
function check_gauge_0R(H::InfiniteCanonicalMPO,H0::InfiniteMPO)::Float64
    eps2=0.0
    for k in eachindex(H)
        eps2+=norm(H.G0R[k - 1] * H.AR[k] - H0[k] * H.G0R[k])^2
    end
    return sqrt(eps2)
end

#-----------------------------------------------------------------------------------
#
# Trivial wrappers: 
#   1) Convert InfiniteMPO->reg_form_iMPO
#   2) Orthogonalize or truncate
#   3) Convert reg_form_iMPO --> InfiniteMPO
#
@doc """
    orthogonalize(H::InfiniteMPO;kwargs...)

Bring `CelledVector` representation of an infinite MPO into left or right canonical form using 
block respecting QR iteration as described in section Vi B and Alogrithm 3 of:
> Daniel E. Parker, Xiangyu Cao, and Michael P. Zaletel Phys. Rev. B 102, 035147
If you intend to also call `truncate!` then do not bother calling `orthogonalize!` beforehand, as `truncate!` will do this automatically and ensure the correct handling of that gauge transforms.

# Arguments
- H::InfiniteMPO which is `CelledVector` of MPO matrices. 

# Keywords
- `rr_cutoff::Float64 = -1.0` : cutoff for rank revealing QR which removes zero pivot rows and columns. All rows with max(abs(R[r,:]))<`rr_cutoff` are considered zero and removed. `rr_cutoff`=-1.0 indicates no rank reduction.
- `eps_qr::Float64 = 1e-13` : Iterations end when ||R-𝕀|| < eps_qr
- `max_iter::Int64=40`: Maximum number if allowed iterations if ||R-𝕀|| does not converge
- `verbose::Bool=false` : Show details of iterations for trouble shooting.

# Returns
- [InfiniteCanonicalMPO](@ref) with AL/AR orthogonalized InfiniteMPOs and required gauge transforms

# Example
```julia
julia> using ITensors, ITensorInfiniteMPS
julia> initstate(n) = "↑";
julia> sites = infsiteinds("Electron", 2;initstate,conserve_qns=true) #Two site unit cell
2-element CelledVector{Index{Vector{Pair{QN, Int64}}}, typeof(translatecelltags)}:
 (dim=4|id=29|"Electron,Site,c=1,n=1") <Out>
 1: QN(("Nf",-1,-1),("Sz",-1)) => 1
 2: QN(("Nf",0,-1),("Sz",0)) => 1
 3: QN(("Nf",0,-1),("Sz",-2)) => 1
 4: QN(("Nf",1,-1),("Sz",-1)) => 1
 (dim=4|id=728|"Electron,Site,c=1,n=2") <Out>
 1: QN(("Nf",-1,-1),("Sz",-1)) => 1
 2: QN(("Nf",0,-1),("Sz",0)) => 1
 3: QN(("Nf",0,-1),("Sz",-2)) => 1
 4: QN(("Nf",1,-1),("Sz",-1)) => 1

 julia> get_Dw(H) #Show bond dimensions of H
2-element Vector{Int64}:
 23
 23

julia> pprint(H[1]) #Schematic view of the operator valued matrix at site 1.
I 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
I 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
S 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
S 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
S 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
S 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
S 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 I 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 S S 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 S 0 S 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 S 0 0 S 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 S 0 0 0 S 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 S 0 0 0 0 S 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 I 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 I 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 S S 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 S 0 S 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 S 0 0 S 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 S 0 0 0 S 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 S 0 0 0 0 S 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 I 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 S S S S S S I I 

julia> Ho = orthogonalize(H); #Rank reducing QR iteration to orthogonal form
julia> check_ortho(Ho.AL,left)
true

julia> check_ortho(Ho.AR,right)
true

julia> [norm(Ho.GLR[k - 1] * Ho.AR[k] - Ho.AL[k] * Ho.GLR[k])^2 for k in 1:2] #test interlaced gauge relations
2-element Vector{Float64}:
 0.000000
 0.000000

 julia> get_Dw(Ho) #Show bond dimensions of Ho
2-element Vector{Int64}:
 17
 17

julia> pprint(Ho.AR[1])
I 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
S S S S 0 0 0 0 0 0 0 0 0 0 0 0 0 
S S S S 0 0 0 0 0 0 0 0 0 0 0 0 0 
S S S S 0 0 0 0 0 0 0 0 0 0 0 0 0 
S 0 0 0 S S S 0 0 0 0 0 0 0 0 0 0 
S 0 0 0 S S S 0 0 0 0 0 0 0 0 0 0 
S 0 0 0 S S S 0 0 0 0 0 0 0 0 0 0 
S 0 0 0 0 0 0 S S S 0 0 0 0 0 0 0 
S 0 0 0 0 0 0 S S S 0 0 0 0 0 0 0 
S 0 0 0 0 0 0 S S S 0 0 0 0 0 0 0 
S 0 0 0 0 0 0 0 0 0 S S S 0 0 0 0 
S 0 0 0 0 0 0 0 0 0 S S S 0 0 0 0 
S 0 0 0 0 0 0 0 0 0 S S S 0 0 0 0 
S 0 0 0 0 0 0 0 0 0 0 0 0 S S S 0 
S 0 0 0 0 0 0 0 0 0 0 0 0 S S S 0 
S 0 0 0 0 0 0 0 0 0 0 0 0 S S S 0 
S S S S S S S S S S S S S S S S I 
```
"""
function ITensors.orthogonalize(H0::InfiniteMPO;kwargs...)::InfiniteCanonicalMPO
    HL,HR,GLR,G0R=orthogonalize(reg_form_iMPO(H0);kwargs...)
    return InfiniteCanonicalMPO(H0,HL,HR,GLR,G0R)
end

@doc """
    truncate!(H::InfiniteMPO;kwargs...)

Truncate a `CelledVector` representation of an infinite MPO as described in section VII and Alogrithm 5 of:
> Daniel E. Parker, Xiangyu Cao, and Michael P. Zaletel Phys. Rev. B 102, 035147
It is not nessecary (or recommended) to call the `orthogonalize` function prior to calling `truncate`. The `truncate` 
function will do this automatically.  This is because the truncation process requires the gauge transform tensors resulting from 
orthogonalizing.  So it is better to do this internally in order to be sure the correct gauge transforms are used.

# Arguments
- `H::InfiniteMPO` which is a `CelledVector` of MPO matrices.

# Keywords
- `rr_cutoff::Float64 = -1.0` : cutoff for rank revealing QR which removes zero pivot rows and columns. 
   All rows with max(abs(R[r,:]))<`rr_cutoff` are considered zero and removed. `rr_cutoff`=1.0 indicate no rank reduction.
- `cutoff::Float64 = 0.0` : Using a `cutoff` allows the SVD algorithm to truncate as many states as possible while still ensuring a certain accuracy. 
- `maxdim::Int64` : If the number of singular values exceeds `maxdim`, only the largest `maxdim` will be retained.
- `mindim::Int64` : At least `mindim` singular values will be retained, even if some fall below the cutoff
   
# Returns
- `InfiniteCanonicalMPO` with HL/HR as truncated `InfiniteMPO`s and the diagonal gauge transforms between HL & HR
- a `bond_spectrums` object which is a `Vector{Spectrum}`

# Example
```julia
julia> using ITensors,ITensorInfiniteMPS

julia> initstate(n) = "↑";

julia> sites = infsiteinds("S=1/2", 2;initstate,conserve_qns=true) #Two site unit cell
2-element CelledVector{Index{Vector{Pair{QN, Int64}}}, typeof(translatecelltags)}:
 (dim=2|id=906|"S=1/2,Site,c=1,n=1") <Out>
 1: QN("Sz",0) => 1
 2: QN("Sz",-2) => 1
 (dim=2|id=995|"S=1/2,Site,c=1,n=2") <Out>
 1: QN("Sz",0) => 1
 2: QN("Sz",-2) => 1

julia> H=InfiniteMPO(Model"heisenbergNNN"(),sites;NNN=3); #Heisenberg model with up to 3rd neighbour interactions

julia> @show get_Dw(H); #Show bond dimensions of H
get_Dw(H) = [17, 17]

julia> pprint(H[1]) #Schematic view of the operator valued matrix at site 1.
I 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
I 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
S 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
S 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
S 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 I 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 S S 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 S 0 S 0 0 0 0 0 0 0 0 0 0 0 0 0 
0 S 0 0 S 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 I 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 I 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 S S 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 S 0 S 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 S 0 0 S 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 I 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 S S S I I 


julia> Ht,bspectrum = truncate(H); #orthogonalize and then truncate. Uses default svd cutoff=1e-15

julia> check_ortho(Ht.AL,left)
true

julia> check_ortho(Ht.AR,right)
true

julia> @show [norm(Ht.GLR[k - 1] * Ht.AR[k] - Ht.AL[k] * Ht.GLR[k])^2 for k in 1:2]; #test interlaced gauge relations
[norm(Ht.GLR[k - 1] * Ht.AR[k] - Ht.AL[k] * Ht.GLR[k]) ^ 2 for k = 1:2] = [3.701212096446601e-31, 3.723257474750299e-31]

julia> @show get_Dw(Ht); #Show bond dimensions of Ho
get_Dw(Ht) = [11, 11]

julia> pprint(Ht.AR[1])
I 0 0 0 0 0 0 0 0 0 0 
S S S S 0 0 0 0 0 0 0 
S S S S 0 0 0 0 0 0 0 
S S S S 0 0 0 0 0 0 0 
S 0 0 0 S S 0 0 0 0 0 
S 0 0 0 S S S 0 0 0 0 
S 0 0 0 S S S 0 0 0 0 
S 0 0 0 0 0 0 S S S 0 
S 0 0 0 0 0 0 S S S 0 
S 0 0 0 0 0 0 S S S 0 
0 S S S S S 0 S S S I 


julia> @show bspectrum;
bspectrum = 
Bond  Ns   max(s)     min(s)    Entropy  Tr. Error
   1    9  0.33333   4.17e-02   0.79862  0.00e+00
   2    9  0.33333   4.17e-02   0.79862  0.00e+00

```
"""
function ITensors.truncate(H0::InfiniteMPO;kwargs...)::Tuple{InfiniteCanonicalMPO,bond_spectrums}
    HL, HR, Ss, Gfull, b_spectrums = truncate!(reg_form_iMPO(H0);kwargs...)
    return InfiniteCanonicalMPO(H0,HL,HR,Ss,Gfull),b_spectrums
end

