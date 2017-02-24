module GMRES
#export naiveGMRES

using DataStructures

function ortho!(h, w, vs)
    orthotol = .1
    initnorm = norm(w)
    for (i,v) = enumerate(vs)
        h[i] = vecdot(w, v)
        LinAlg.axpy!(-h[i],v,w)
    end
    finalnorm = norm(w)
    normratio = finalnorm / initnorm
    if normratio < orthotol
        warn("Orthoganalization cancellation failure, normratio $normratio, reorthogonalizing")
        for (i,v) = enumerate(vs)
            f = vecdot(w, v)
            LinAlg.axpy!(-f,v,w)
            h[i] += f
        end        
    end
    w
end

function naiveGMRES(A,b;
                    numiters=10,
                    numvecs=numiters,
                    diom=false,
                    tol=sqrt(eps(eltype(A))))
    dim = LinAlg.checksquare(A)
    x = zeros(dim)
    β = norm(b)
    V = zeros(eltype(A),dim,numiters+1)
    V[:,1] = b/β
    H = zeros(eltype(A),numiters+1,numiters)
    for i = 1:numiters
        s = max(i-numvecs+1,1)
        w=ortho!(@view(H[s:end,i]),
                 A*V[:,i],
                 (V[:,x] for x = s:i))
        H[i+1,i] = norm(w)
        if H[i+1,i] < tol
            warn("tol fail")
            break;
        end
        V[:,i+1]=w/H[i+1,i]
    end
    if !diom
        (Q,R) = qr(H,Val{false},thin=false)
        g = At_mul_B(Q,[β;zeros(numiters)])
    else
        (Q,R) = qr(@view(H[1:end-1,:]),Val{false},thin=false)
        g = At_mul_B(Q,[β;zeros(numiters-1)])
    end
    y = R \ @view(g[1:numiters])
    x = @view(V[:,1:numiters]) * y
    x,g,V,H,Q,R,y
end

function incompleteGMRES(A,b, x=zeros(b);
                         maxiters=100,
                         numvecs=10,
                         tol=sqrt(eps(eltype(A))))
    dim = size(b)[1]
    r₀ = b - A*x
    β = norm(r₀)
    V = zeros(eltype(A),dim,maxiters+1)
    V[:,1] = r₀/β
    H = zeros(eltype(A),maxiters+1,maxiters)
    g = [β ; zeros(maxiters)]
    Q = LinAlg.Rotation{eltype(A)}([])
    R = zeros(eltype(A),maxiters,maxiters)
    for i = 1:maxiters
        s = max(i-numvecs + 1, 1)
        w=ortho!(@view(H[s:end,i]),
                 A*V[:,i],
                 (V[:,x] for x = s:i))
        H[i+1,i] = norm(w)
        if H[i+1,i] < tol
            error("tol fail")
        end
        V[:,i+1]=w/H[i+1,i]
        R[1:i,i] = H[1:i,i]
        for Ωₙ = Q.rotations
            A_mul_B!(Ωₙ,view(R,:,i))
        end
        (Ω,h) = givens(R[i,i],H[i+1,i],i,i+1)
        R[i,i] = h
        A_mul_B!(Ω, g)
        A_mul_B!(Ω, Q)
        if abs(g[i+1]) < tol
            R = view(R,1:i,1:i)
            g = view(g,1:i+1)
            V = view(V,:,1:i+1)
            break
        end
    end
    y = R\@view(g[1:end-1])
    x = @view(V[:,1:end-1]) * y
    x,g,V,H,Q,R,y
end

@inline function safepush!(C,x)
    if capacity(C) == length(C)
        shift!(C)
    end
    push!(C,x)
end

function inneroptGMRES(A,b,x,maxiters,numvecs,tol)
    dim = size(b)[1]
    T = eltype(b)

    local r₀,r,V,h,Q,γ,P,stallp
    initp = true
    
    for i = 1:maxiters

        if initp
            r₀ = b - A*x
            r = β = norm(r₀)
            V = CircularDeque{Vector{T}}(numvecs)
            push!(V,r₀/β)
            h = zeros(T,numvecs+2)
            Q = CircularDeque{LinAlg.Givens{T}}(numvecs)
            γ = [β ; 0]
            P = CircularDeque{Vector{T}}(numvecs)
            stallp = false
            initp = false
        end
        
        len = length(V)
        qs = length(Q)
        hoff = qs < numvecs ? 0 : 1

        # Orthogonalize
        h[1]=0
        w=ortho!(view(h,1+hoff:len+hoff), A*back(V), V)
        h[len+1+hoff] = ω = norm(w)

        # Apply rotations to h
        for j = 1:qs
            A_mul_B!(Q[j],view(h,j:j+1))
        end
        (Ω,h[len+hoff]) =
            givens(h[len+hoff],h[len+1+hoff],1,2)
        h[len+1+hoff]=0
        A_mul_B!(Ω,γ)
        safepush!(Q, Ω)

        # Next p
        p = copy(back(V))
        for j = 1:length(P)
            LinAlg.axpy!(-h[j],P[j],p)
        end
        scale!(p,one(h[1])/h[length(P)+1])
        safepush!(P,p)

        # Update x
        LinAlg.axpy!(γ[1],p,x)

        # Termination
        if abs(γ[2]) < tol
            @goto success
        end

        # Check for stalls
        if isapprox(r, abs(γ[2]))
            if stallp
                warn("Stalled at $(abs(γ[2]))")
                initp = true
                continue
            else
                stallp = true
            end
        end
        r = abs(γ[2])

        # Ready next iteration
        γ = [ γ[2] ; 0 ]        
        if ω < tol
            error("ortho failure")
        end
        safepush!(V,w/ω)
    end
    warn("Terminated at maxiter, $(abs(γ[1]))")

    @label success
    x
end

function optGMRES(A,b, x=zeros(b);
                  maxiters::Int=100,
                  numvecs::Int=10,
                  tol::Float64=sqrt(eps(eltype(b))))
    inneroptGMRES(A,b,x,maxiters,numvecs,tol)
end

cleanmat(A) = map(x -> abs(x) < sqrt(eps(eltype(A))) ? 0 : x, A)

end
