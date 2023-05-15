using ITensors
using ITensorMPOCompression
using ITensorInfiniteMPS
using Revise

import ITensorInfiniteMPS: reg_form_iMPO, is_gauge_fixed, gauge_fix!

N = 6
model = Model"fqhe_2b_pot"()
model_params = (Vs= [1.0, 1, 0.1], Ly = 6.0, prec = 1e-8)
function initstate(n)
	return (n%3 == 0) ? 2 : 1
end
p = 1
q = 3
conserve_momentum = true


function fermion_momentum_translator(i::Index, n::Integer; N=N)
  ts = tags(i)
  translated_ts = ITensorInfiniteMPS.translatecelltags(ts, n)
  new_i = replacetags(i, ts => translated_ts)
  for j in 1:length(new_i.space)
    ch = new_i.space[j][1][1].val
    mom = new_i.space[j][1][2].val
    new_i.space[j] = Pair(QN(("Nf", ch), ("NfMom", mom + n * N * ch)), new_i.space[j][2])
  end
  return new_i
end

function ITensors.space(::SiteType"FermionK", pos::Int; p=1, q=1, conserve_momentum=true)
  if !conserve_momentum
    return [QN("Nf", -p) => 1, QN("Nf", q - p) => 1]
  else
    return [
      QN(("Nf", -p), ("NfMom", -p * pos)) => 1,
      QN(("Nf", q - p), ("NfMom", (q - p) * pos)) => 1,
    ]
  end
end

function ITensors.op!(Op::ITensor, opname::OpName, ::SiteType"FermionK", s::Index...)
  return ITensors.op!(Op, opname, SiteType("Fermion"), s...)
end


s = infsiteinds(
  "FermionK", N; translator=fermion_momentum_translator, initstate, conserve_momentum, p, q
);

ψ = InfMPS(s, initstate);

Hm = InfiniteMPOMatrix(model, s; model_params...);
for H in Hm
  lx,ly=size(H)
  for ix in 1:lx
    for iy in 1:ly
      M=H[ix,iy]
      if !isempty(M)
        ITensors.checkflux(M)
      end
    end
  end
end

Hi=InfiniteMPO(Hm)

ITensors.checkflux.(Hi)

@show get_Dw(Hi)
Hc=orthogonalize(Hi)
@show get_Dw(Hc.AL)
Ht,BondSpectrums = truncate(Hi)
@show get_Dw(Ht.AL)
@show BondSpectrums
Ht,BondSpectrums = truncate(Hi;cutoff=1e-10)
@show get_Dw(Ht.AL)
@show BondSpectrums

nothing
