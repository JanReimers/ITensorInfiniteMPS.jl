using ITensors,ITensorInfiniteMPS
initstate(n) = "↑";
sites = infsiteinds("S=1/2", 2;initstate,conserve_qns=true) #Two site unit cell
H=InfiniteMPO(Model"heisenbergNNN"(),sites;NNN=3); #Heisenberg model with up to 3rd neighbour interactions
@show get_Dw(H); #Show bond dimensions of H
pprint(H[1]) #Schematic view of the operator valued matrix at site 1.
Ht,bspectrum = truncate(H); #orthogonalize and then truncate. Uses default svd cutoff=1e-15
check_ortho(Ht.AL,left)
check_ortho(Ht.AR,right)
@show [norm(Ht.GLR[k - 1] * Ht.AR[k] - Ht.AL[k] * Ht.GLR[k])^2 for k in 1:2]; #test interlaced gauge relations
@show get_Dw(Ht); #Show bond dimensions of Ho
pprint(Ht.AR[1])
@show bspectrum;