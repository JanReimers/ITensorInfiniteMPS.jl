import ITensorMPOCompression: slice, parse_links, parse_site

function translatecell(translator::Function, V::Vector{ITensor}, n::Integer)
    if n==0
        return V
    end
    V1=Vector{ITensor}()
    for T in V
        if length(inds(T))>0
            # @show n T translator
            Tt=translatecell(translatecelltags, T, n)
            # @show Tt
            push!(V1,Tt)
        else
            push!(V1,T)
        end
    end
    return V1
end


struct ALk
    ψ::InfiniteCanonicalMPS
    k::Int
end

function (A::ALk)(x)
    ψ = A.ψ
    ψ′ =dag(ψ)'
    k = A.k
    N = nsites(ψ)
    l = linkinds(only, ψ.AL)
    # l′ = linkinds(only, ψ′.AL)
    r = linkinds(only, ψ.AR)
    # r′ = linkinds(only, ψ′.AR)
    s = siteinds(only, ψ)
    δʳ(kk) = δ(dag(r[kk]), prime(r[kk]))
    δˡ(kk) = δ(dag(l[kk]), prime(l[kk]))
    # δˡ(kk) = δ(l[kk], l′[kk])
    δˢ(kk) = δ(dag(s[kk]), prime(s[kk]))
    xT = translatecell(translator(ψ), x, -1) #xT = x shifted one unit cell to the left.
    for k′=k-N+1:k
        xT*=ψ′.AL[k′]*δˢ(k′)*ψ.AL[k′]
        # @show ψ′.AL[k′]*δˢ(k′)*ψ.AL[k′]
    end
    R=ψ.C[k] * (ψ′.C[k] * δʳ(k))
    𝕀=denseblocks(δˡ(k))
    xR=x*R*𝕀
    # @show xT xR
    return xT-xR
end

struct ARk
    ψ::InfiniteCanonicalMPS
    k::Int
end

function (A::ARk)(x)
    ψ = A.ψ
    ψ′ =dag(ψ)'
    k = A.k
    N = nsites(ψ)
    l = linkinds(only, ψ.AL)
    # l′ = linkinds(only, ψ′.AL)
    r = linkinds(only, ψ.AR)
    # r′ = linkinds(only, ψ′.AR)
    s = siteinds(only, ψ)
    δʳ(kk) = δ(dag(r[kk]), prime(r[kk]))
    δˡ(kk) = δ(dag(l[kk]), prime(l[kk]))
    # δˡ(kk) = δ(l[kk], l′[kk])
    δˢ(kk) = δ(dag(s[kk]), prime(s[kk]))

    # @show inds(x)
    xT = translatecell(translator(ψ), x, 1) #xT = x shifted one unit cell to the right.
    # @show inds(xT)
    
    # TL=nothing
    for k′=k+N:-1:k+1
        # @show k k′ inds(ψ.AR[k′])
        xT*=ψ′.AR[k′]*δˢ(k′)*ψ.AR[k′]
        @assert order(xT)==2
        # if isnothing(TL)
        #     TL=ψ′.AR[k′]*δˢ(k′)*ψ.AR[k′]
        # else
        #     TL*=ψ′.AR[k′]*δˢ(k′)*ψ.AR[k′]
        # end
    end
    L=ψ.C[k] * (ψ′.C[k]*δˡ(k))
    𝕀=denseblocks(δʳ(k))

    xL=𝕀*L*x # |𝕀)(L|x)
    #@show TL 𝕀*L

    return xT-xL
end

#
#   (Lₖ|=(Lₖ₋₁|*T(k)ᵂₗ
#
function apply_TW_left(Lₖ₋₁::Vector{ITensor},Ŵ::ITensor,ψ::ITensor,δˢk::ITensor;skip_𝕀=false)
    ψ′ =dag(ψ)'
    il,ir=parse_links(Ŵ)
    dh,_,_=parse_site(Ŵ)
    @assert dim(il)==dim(ir)
    Dw=dim(il)
    Lₖ=Vector{ITensor}(undef,Dw)
    for b in Dw:-1:1
        for a in Dw:-1:b
            if isassigned(Lₖ₋₁,a) 
                Wab=slice(Ŵ,il=>a,ir=>b)
                is_zero=norm(Wab)==0.0
                is_𝕀=!is_zero && scalar(Wab*dag(δˢk))==dh
                is_diag= a==b
                if !is_zero && is_diag && a>1 && a<Dw
                    if is_𝕀
                        @error "apply_TW_left: found unit operator on the diagonal, away from the corners"
                    else
                        @error "apply_TW_left: found non-zero operator on the diagonal, away from the corners. This is not supported yet"
                    end
                    @show a b Wab
                    @assert false
                end
                if skip_𝕀 && is_diag && is_𝕀 && a==1 
                    println("Skipping unit op")
                    continue #skip the 𝕀 op in the upper left corner
                end
                if !isassigned(Lₖ,b)
                    Lₖ[b]=emptyITensor()
                end
                Lₖ[b]+=Lₖ₋₁[a]*ψ′*Wab*ψ
                @assert order(Lₖ[b])==2
                
            end # if L[a] assigned
        end # for a
    end # for b
    return Lₖ
end


function left_environment(H::InfiniteMPO, ψ::InfiniteCanonicalMPS; tol=1e-10)
    ψ′ =dag(ψ)'
    l = linkinds(only, ψ.AL)
    r = linkinds(only, ψ.AR)
    s = siteinds(only, ψ)
    δʳ(kk) = δ(dag(r[kk]), prime(r[kk]))
    δˡ(kk) = δ(dag(l[kk]), prime(l[kk]))
    δˢ(kk) = δ(dag(s[kk]), prime(s[kk]))
    D=dim(l[1])
    # dh=dim(s[1])
    il,ir=ITensorMPOCompression.parse_links(H[1])
    @assert dim(il)==dim(ir)
    Dw=dim(il)
    N=nsites(ψ)
    # Lₖ₋₁=Vector{ITensor}(undef,Dw)
    L₁=Vector{ITensor}(undef,Dw)
    
    #
    #  Solve for k=1
    #
    L₁[Dw]=δˡ(1) #Left eigen vector of TL (not TWL)
    for b1 in Dw-1:-1:1 #sweep backwards from Dw-1 down.
        #
        #  Load up all the know tensors from b1+1 to Dw.  Also translate one unit to the left.
        #
        Lₖ₋₁=Vector{ITensor}(undef,Dw)
        for b in Dw:-1:b1+1
            @assert isassigned(L₁,b) 
            Lₖ₋₁[b]=translatecell(translator(ψ), L₁[b], -1)
        end
        #
        #  Loop throught the unit cell and apply Tᵂₗ
        #
        for k in 2-N:1
            Lₖ₋₁=apply_TW_left(Lₖ₋₁,H[k],ψ.AL[k],δˢ(k);skip_𝕀=false)
        end # for k
        @assert isassigned(Lₖ₋₁,b1) 
        L₁[b1]=Lₖ₋₁[b1] #save the new value.
    end #for b1
    # println("Done L1")

    # @show array.(L₁) inds.(L₁)
    localR = ψ.C[1] * δʳ(1) * ψ′.C[1] #to revise
    eₗ = [0.0]
    eₗ[1] = (L₁[1] * localR)[]
    # @show localR array.(L₁) eₗ[1]
    L₁[1] += -(eₗ[1] * denseblocks(δˡ(1))) #from Loic's MPOMatrix code.
    A = ALk(ψ, 1)
    L₁[1], info = linsolve(A, L₁[1], 1, -1; tol=tol)

    vN=[[emptyITensor() for n in 1:Dw] for n in 1:N]
    L=CelledVector{Vector{ITensor}}(vN,translatecell)
    L[1]=L₁

    #
    #  Now sweep throught the cell and evlaute all the L[k] form L[1]
    #
    for k in 2:N
        L[k]=apply_TW_left(L[k-1],H[k],ψ.AL[k],δˢ(k))
    end
    #
    #  Verify that we get L[1] back from L[0]
    #
    L₁=apply_TW_left(L[0],H[1],ψ.AL[1],δˢ(1))
    for b in 2:Dw #We know that L₁[1] is wrong
        if norm(L₁[b]-L[1][b])>1e-15*D*N
            @show L₁[b] L[1][b]
            @assert  false
        end
    end

    return L,eₗ[1]
end

#
#   |Rₖ₋₁)=T(k)ᵂᵣ*|Rₖ)
#
function apply_TW_right(Rₖ::Vector{ITensor},Ŵ::ITensor,ψ::ITensor,δˢk::ITensor;skip_𝕀=false)
    ψ′ =dag(ψ)'
    il,ir=parse_links(Ŵ)
    dh,_,_=parse_site(Ŵ)
    @assert dim(il)==dim(ir)
    Dw=dim(il)
    Rₖ₋₁=Vector{ITensor}(undef,Dw)
    for a in 1:Dw
        for b in 1:Dw
            if isassigned(Rₖ,b) 
                Wab=slice(Ŵ,il=>a,ir=>b)
                is_zero=norm(Wab)==0.0
                is_𝕀=!is_zero && scalar(Wab*dag(δˢk))==dh
                is_diag= a==b
                if !is_zero && is_diag && b>1 && b<Dw
                    if is_𝕀
                        @error "apply_TW_left: found unit operator on the diagonal, away from the corners"
                    else
                        @error "apply_TW_left: found non-zero operator on the diagonal, away from the corners. This is not supported yet"
                    end
                    @show a b Wab
                    @assert false
                end
                if skip_𝕀 && is_diag && is_𝕀 && a==Dw 
                    println("Skipping unit op")
                    continue #skip the 𝕀 op in the lower right corner
                end
                if !isassigned(Rₖ₋₁,a)
                    Rₖ₋₁[a]=emptyITensor()
                end
                # @show Rₖ₋₁[b]
                Rₖ₋₁[a]+=ψ′*Wab*ψ*Rₖ[b]
                # @show   ψ′*Wab*ψ Rₖ[a]
                @assert order(Rₖ₋₁[a])==2
                
            end # if R[b] assigned
        end # for b
    end # for a
    return Rₖ₋₁
end

function right_environment(H::InfiniteMPO, ψ::InfiniteCanonicalMPS; tol=1e-10)
    ψ′ =dag(ψ)'
    l = linkinds(only, ψ.AL)
    r = linkinds(only, ψ.AR)
    s = siteinds(only, ψ)
    δʳ(kk) = δ(dag(r[kk]), prime(r[kk]))
    δˡ(kk) = δ(dag(l[kk]), prime(l[kk]))
    δˢ(kk) = δ(dag(s[kk]), prime(s[kk]))
    D=dim(l[1])
    il,ir=ITensorMPOCompression.parse_links(H[1])
    @assert dim(il)==dim(ir)
    Dw=dim(il)
    N=nsites(ψ)
    R₁=Vector{ITensor}(undef,Dw)
    
    #
    #  Solve for k=1
    #
    R₁[1]=δʳ(1) #right eigen vector of TR 
    for b1 in 2:Dw 
        #
        #  Load up all the know tensors from 1 to b1-1.  Also translate one unit to the right.
        #
        Rₖ=Vector{ITensor}(undef,Dw)
        for b in 1:b1-1
            @assert isassigned(R₁,b) 
            Rₖ[b]=translatecell(translator(ψ), R₁[b], 1)
        end
        #
        #  Loop throught the unit cell and apply Tᵂₗ
        #
        for k in N+1:-1:2
            Rₖ=apply_TW_right(Rₖ,H[k],ψ.AR[k],δˢ(k);skip_𝕀=false)
        end # for k
        @assert isassigned(Rₖ,b1) 
        R₁[b1]=Rₖ[b1] #save the new value.
    end #for b1
    
    localL = ψ.C[1] * δˡ(1) * dag(prime(ψ.C[1]))
    eᵣ=[0.0]
    eᵣ[1] = (localL * R₁[Dw])[]
    # @show localL array.(R₁) eᵣ[1]
    R₁[Dw] += -(eᵣ[1] * denseblocks(δʳ(1)))
    A = ARk(ψ, 1)
    R₁[Dw], info = linsolve(A, R₁[Dw], 1, -1; tol=tol)

    vN=[[emptyITensor() for n in 1:Dw] for n in 1:N]
    R=CelledVector{Vector{ITensor}}(vN,translatecell)
    R[1]=R₁

    #
    #  Now sweep leftwards through the cell and evalaute all the R[k] form R[1]
    #
    for k in N:-1:2
        R[k]=apply_TW_right(R[k+1],H[k+1],ψ.AR[k+1],δˢ(k+1))
    end
    #
    #  Verify that we get R[1] back from R[2]
    #
    # @show inds.(R[2])  inds(ψ.AR[2]) inds(H[2])
    R₁=apply_TW_right(R[2],H[2],ψ.AR[2],δˢ(2))
    for b in 1:Dw-1 #We know that R₁[Dw] is wrong
        if norm(R₁[b]-R[1][b])>1e-15*D*N
            @show R₁[b] R[1][b]
            @assert  false
        end
    end

    return R,eᵣ[1]
end

struct iMPO¹
    L::Vector{ITensor}
    R::Vector{ITensor}
    Ŵ::ITensor
  end
  
  function (H::iMPO¹)(x)
    L = H.L
    R = H.R
    Ŵ = H.Ŵ
    Dw = length(L)
    il,ir=parse_links(Ŵ)
    # @show il ir inds(Ŵ)
    @assert dim(il)==Dw
    @assert dim(ir)==Dw
    result = ITensor(prime(inds(x)))
    for a in 1:Dw
      for b in 1:Dw
        Wab=slice(Ŵ,il=>a,ir=>b)
        if norm(Wab)>0.0
          result += L[a] * x * Wab * R[b]
        end
      end
    end
    return noprime(result)
end

function tdvp_iteration_sequential(
    solver::Function,
    H::InfiniteMPO,
    ψ::InfiniteCanonicalMPS;
    (ϵᴸ!)=fill(1e-15, nsites(ψ)),
    (ϵᴿ!)=fill(1e-15, nsites(ψ)),
    time_step,
    solver_tol=(x -> x / 100),
    eager=true,
  )
    ψ = copy(ψ)
    ϵᵖʳᵉˢ = max(maximum(ϵᴸ!), maximum(ϵᴿ!))
    _solver_tol = solver_tol(ϵᵖʳᵉˢ)
    N = nsites(ψ)
  
    C̃ = InfiniteMPS(Vector{ITensor}(undef, N))
    Ãᶜ = InfiniteMPS(Vector{ITensor}(undef, N))
    Ãᴸ = InfiniteMPS(Vector{ITensor}(undef, N))
    Ãᴿ = InfiniteMPS(Vector{ITensor}(undef, N))
  
    eL = zeros(N)
    eR = zeros(N)
    for n in 1:N
      L, eL[n] = left_environment(H, ψ; tol=_solver_tol) #TODO currently computing two many of them
      R, eR[n] = right_environment(H, ψ; tol=_solver_tol) #TODO currently computing two many of them
      if N == 1
        # 0-site effective Hamiltonian
        E0, C̃[n], info0 = solver(H⁰(L[1], R[1]), time_step, ψ.C[1], _solver_tol, eager)
        # 1-site effective Hamiltonian
        E1, Ãᶜ[n], info1 = solver(
            iMPO¹(L[0], R[1], H[1]), time_step, ψ.AL[1] * ψ.C[1], _solver_tol, eager
        )
        Ãᴸ[1] = ortho_polar(Ãᶜ[1], C̃[1])
        Ãᴿ[1] = ortho_polar(Ãᶜ[1], C̃[0])
        ψ.AL[1] = Ãᴸ[1]
        ψ.AR[1] = Ãᴿ[1]
        ψ.C[1] = C̃[1]
      else
        # @show n L[n] R[n+1]
        # 0-site effective Hamiltonian
        E0, C̃[n], info0 = solver(H⁰(L[n], R[n]), time_step, ψ.C[n], _solver_tol, eager)
        E0′, C̃[n - 1], info0′ = solver(
          H⁰(L[n - 1], R[n-1]), time_step, ψ.C[n - 1], _solver_tol, eager
        )
        # 1-site effective Hamiltonian
        E1, Ãᶜ[n], info1 = solver(
            iMPO¹(L[n - 1], R[n ], H[n]), time_step, ψ.AL[n] * ψ.C[n], _solver_tol, eager
        )
        Ãᴸ[n] = ortho_polar(Ãᶜ[n], C̃[n])
        Ãᴿ[n] = ortho_polar(Ãᶜ[n], C̃[n - 1])
        ψ.AL[n] = Ãᴸ[n]
        ψ.AR[n] = Ãᴿ[n]
        ψ.C[n] = C̃[n]
        ψ.C[n - 1] = C̃[n - 1]
      end
    end
    for n in 1:N
      ϵᴸ![n] = norm(Ãᶜ[n] - Ãᴸ[n] * C̃[n])
      ϵᴿ![n] = norm(Ãᶜ[n] - C̃[n - 1] * Ãᴿ[n])
    end
    return ψ, (eL / N, eR / N)
end
  
function tdvp_iteration_parallel(
    solver::Function,
    H::InfiniteMPO,
    ψ::InfiniteCanonicalMPS;
    (ϵᴸ!)=fill(1e-15, nsites(ψ)),
    (ϵᴿ!)=fill(1e-15, nsites(ψ)),
    time_step,
    solver_tol=(x -> x / 100),
    eager=true,
  )
    ψ = copy(ψ)
    ϵᵖʳᵉˢ = max(maximum(ϵᴸ!), maximum(ϵᴿ!))
    _solver_tol = solver_tol(ϵᵖʳᵉˢ)
    N = nsites(ψ)
  
    C̃ = InfiniteMPS(Vector{ITensor}(undef, N))
    Ãᶜ = InfiniteMPS(Vector{ITensor}(undef, N))
    Ãᴸ = InfiniteMPS(Vector{ITensor}(undef, N))
    Ãᴿ = InfiniteMPS(Vector{ITensor}(undef, N))
  
    eL = zeros(1)
    eR = zeros(1)
    L, eL[1] = left_environment(H, ψ; tol=_solver_tol) #TODO currently computing two many of them
    R, eR[1] = right_environment(H, ψ; tol=_solver_tol) #TODO currently computing two many of them
    for n in 1:N
      if N == 1
        # 0-site effective Hamiltonian
        E0, C̃[n], info0 = solver(H⁰(L[1], R[2]), time_step, ψ.C[1], _solver_tol, eager)
        # 1-site effective Hamiltonian
        E1, Ãᶜ[n], info1 = solver(
          H¹(L[0], R[2], H[1]), time_step, ψ.AL[1] * ψ.C[1], _solver_tol, eager
        )
        Ãᴸ[1] = ortho_polar(Ãᶜ[1], C̃[1])
        Ãᴿ[1] = ortho_polar(Ãᶜ[1], C̃[0])
        ψ.AL[1] = Ãᴸ[1]
        ψ.AR[1] = Ãᴿ[1]
        ψ.C[1] = C̃[1]
      else
        # 0-site effective Hamiltonian
        for n in 1:N
          E0, C̃[n], info0 = solver(H⁰(L[n], R[n + 1]), time_step, ψ.C[n], _solver_tol, eager)
          E1, Ãᶜ[n], info1 = solver(
            H¹(L[n - 1], R[n + 1], H[n]), time_step, ψ.AL[n] * ψ.C[n], _solver_tol, eager
          )
        end
        # 1-site effective Hamiltonian
        for n in 1:N
          Ãᴸ[n] = ortho_polar(Ãᶜ[n], C̃[n])
          Ãᴿ[n] = ortho_polar(Ãᶜ[n], C̃[n - 1])
          ψ.AL[n] = Ãᴸ[n]
          ψ.AR[n] = Ãᴿ[n]
          ψ.C[n] = C̃[n]
        end
      end
    end
    for n in 1:N
      ϵᴸ![n] = norm(Ãᶜ[n] - Ãᴸ[n] * C̃[n])
      ϵᴿ![n] = norm(Ãᶜ[n] - C̃[n - 1] * Ãᴿ[n])
    end
    return ψ, (eL / N, eR / N)
end