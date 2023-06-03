# H = Σⱼ (½ S⁺ⱼS⁻ⱼ₊₁ + ½ S⁻ⱼS⁺ⱼ₊₁ + SᶻⱼSᶻⱼ₊₁)
function unit_cell_terms(::Model"heisenberg")
  opsum = OpSum()
  opsum += 0.5, "S+", 1, "S-", 2
  opsum += 0.5, "S-", 1, "S+", 2
  opsum += "Sz", 1, "Sz", 2
  return opsum
end

#  
#  Useful for testing more complex MPOs with distant neightbour interactions.
#  H = ΣⱼΣₙ Jₙ(½ S⁺ⱼS⁻ⱼ₊ₙ + ½ S⁻ⱼS⁺ⱼ₊ₙ + SᶻⱼSᶻⱼ₊ₙ)
#     Jₙ = 1/n decaying interaction strength
#     NNN = Number of Nearest Neightbours
#
function ITensorInfiniteMPS.unit_cell_terms(::Model"heisenbergNNN"; NNN::Int64)
  opsum = OpSum()
  for n in 1:NNN
    J = 1.0 / n
    opsum += J * 0.5, "S+", 1, "S-", 1 + n
    opsum += J * 0.5, "S-", 1, "S+", 1 + n
    opsum += J, "Sz", 1, "Sz", 1 + n
  end
  return opsum
end

"""
    reference(::Model"heisenberg", ::Observable"energy"; N)

Compute the analytic isotropic heisenberg chain ground energy per site for length `N`.
Assumes the heisenberg model is defined with spin
operators not pauli matrices (overall factor of 2 smaller). Taken from [1].

[1] Nickel, Bernie. "Scaling corrections to the ground state energy
of the spin-½ isotropic anti-ferromagnetic Heisenberg chain." Journal of
Physics Communications 1.5 (2017): 055021
"""
function reference(::Model"heisenberg", ::Observable"energy"; N=∞)
  isinf(N) && return (0.5 - 2 * log(2)) / 2
  E∞ = (0.5 - 2 * log(2)) * N
  Eᶠⁱⁿⁱᵗᵉ = π^2 / (6N)
  correction = 1 + 0.375 / log(N)^3
  return (E∞ - Eᶠⁱⁿⁱᵗᵉ * correction) / (2N)
end
