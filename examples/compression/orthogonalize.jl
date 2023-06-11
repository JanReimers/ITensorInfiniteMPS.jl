using ITensors,ITensorInfiniteMPS
initstate(n) = "↑";
sites = infsiteinds("Electron", 2;initstate,conserve_qns=true) #Two site unit cell
H=InfiniteMPO(Model"hubbardNNN"(),sites;NNN=3); #Hubbard model with up to 3rd neighbour interactions
get_Dw(H) #Show bond dimensions of H
pprint(H[1]) #Schematic view of the operator valued matrix at site 1.
Ho = orthogonalize(H); #Rank reducing QR iteration to orthogonal form
check_ortho(Ho.AL,left)
check_ortho(Ho.AR,right)
[norm(Ho.GLR[k - 1] * Ho.AR[k] - Ho.AL[k] * Ho.GLR[k])^2 for k in 1:2] #test interlaced gauge relations
get_Dw(Ho) #Show bond dimensions of Ho
pprint(Ho.AR[1])

