module ITensorInfiniteMPS

# For `only`, which was introduced in Julia v1.4
using Compat
using ITensors
using ITensorMPOCompression
# For optional ITensorsVisualization dependency.
using Requires
# For using ∞ as lengths, ranges, etc.
using Infinities
# For functions like `isdiag`
using LinearAlgebra
# For indexing starting from something other than 0.
using OffsetArrays
using IterTools
# For HDF5 support
using HDF5
# For integration support when computing exact reference results
using QuadGK
# For `groupreduce`, used when splitting up an OpSum
# with the unit cell terms into terms acting on each
# site.
using SplitApplyCombine

using ITensors.NDTensors: eachdiagblock
using KrylovKit: eigsolve, linsolve, exponentiate

import Base: getindex, length, setindex!, +, -, *, truncate

import ITensors: AbstractMPS, ⊕, permute, setinds

import ITensorMPOCompression: @mpoc_assert, parse_links, slice, assign!
import ITensorMPOCompression: orth_type, left, right, reg_form, lower, upper, reg_form_Op, is_regular_form, check
import ITensorMPOCompression: regform_blocks, extract_blocks,  A0, b0, c0, vector_o2, set_𝐛̂_block!, set_𝐜̂_block!, set_𝐝̂_block!
import ITensorMPOCompression: gauge_fix!,is_gauge_fixed, ac_qx, forward, redim, grow, sweep, check_ortho



include("ITensors.jl")
include("ITensorNetworks.jl")
include("itensormap.jl")
include("celledvectors.jl")
include("abstractinfinitemps.jl")
include("infinitemps.jl")
include("infinitempo.jl")
include("infinitecanonicalmps.jl")
include("iMPO_compression/reg_form_iMPO.jl")
include("infinitecanonicalmpo.jl")
include("infinitempomatrix.jl")
include("transfermatrix.jl")
include("models/models.jl")
include("models/fqhe13.jl")
include("models/ising.jl")
include("models/ising_extended.jl")
include("models/heisenberg.jl")
include("models/hubbard.jl")
include("models/xx.jl")
include("orthogonalize.jl")
include("infinitemps_approx.jl")
include("subspace_expansion.jl")
include("vumps_generic.jl")
include("vumps_localham.jl")
include("vumps_nonlocalham.jl")
include("vumps_mpo.jl")
include("iMPO_compression/gauge_fix.jl")
include("iMPO_compression/orthogonalize.jl")
include("iMPO_compression/truncate.jl")
include("vumps_impo.jl")

export Cell,
  CelledVector,
  InfiniteMPS,
  InfiniteCanonicalMPO,
  InfiniteCanonicalMPS,
  InfMPS,
  InfiniteSum,
  InfiniteMPO,
  InfiniteMPOMatrix,
  InfiniteSumLocalOps,
  ITensorMap,
  ITensorNetwork,
  TransferMatrix,
  @Model_str,
  Model,
  @Observable_str,
  Observable,
  check_gauge_LR, 
  check_gauge_0R, 
  check_ortho,
  get_Dw,
  infinitemps_approx,
  infsiteinds,
  input_inds,
  left,
  lower,
  nsites,
  orthogonalize,
  orth_type,
  output_inds,
  pprint,
  reference,
  regform_blocks,
  right,
  subspace_expansion,
  translatecell,
  translatecelltags,
  translator,
  truncate,
  tdvp,
  upper,
  vumps,
  finite_mps,
  ⊕,
  ⊗,
  ×

function __init__()
  # This is used for debugging using visualizations
  @require ITensorsVisualization = "f2aed53d-2f32-47c3-a7b9-1ee253853786" @eval using ITensorsVisualization
end

end
