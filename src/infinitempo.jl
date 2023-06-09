
#
# InfiniteMPO
#

mutable struct InfiniteMPO <: AbstractInfiniteMPS
  data::CelledVector{ITensor}
  llim::Int #RealInfinity
  rlim::Int #RealInfinity
  reverse::Bool
end

translator(mpo::InfiniteMPO) = mpo.data.translator

InfiniteMPO(data::Vector{ITensor},translator::Function) = InfiniteMPO(CelledVector{ITensor}(data,translator), 0, size(data, 1), false)
InfiniteMPO(data::CelledVector{ITensor}) = InfiniteMPO(data, 0, size(data, 1), false)

function ITensorMPOCompression.get_Dw(H::InfiniteMPO)::Vector{Int64}
  N = length(H)
  Dws = Vector{Int64}(undef, N )
  for n in 1:N
    l = commonind(H[n], H[n + 1])
    Dws[n] = dim(l)
  end
  return Dws
end



