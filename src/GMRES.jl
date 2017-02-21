module GMRES
export naiveGMRES

function ortho!(h, w, vs)
    orthotol = .1
    initnorm = norm(w)
    for (i,v) = enumerate(vs)
        h[i] = dot(w, v)
        LinAlg.axpy!(-h[i],v,w)
    end
    finalnorm = norm(w)
    normratio = finalnorm / initnorm
    if normratio < orthotol
        warn("Orthoganalization cancellation failure, normratio $normratio, reorthogonalizing")
        for (i,v) = enumerate(vs)
            f = dot(w, v)
            LinAlg.axpy!(-f,v,w)
            h[i] += f
        end        
    end
    w
end

function naiveGMRES(A,b;
                    numiters=10,
                    tol=sqrt(eps(eltype(A))))
    dim = LinAlg.checksquare(A)
    x = zeros(dim)
    β = norm(b)
    V = zeros(eltype(A),dim,numiters+1)
    V[:,1] = b/β
    H = zeros(eltype(A),numiters+1,numiters)
    for i = 1:numiters
        w=ortho!(view(H,:,i),
                 A*V[:,i],
                 (V[:,x] for x = 1:i))
        H[i+1,i] = norm(w)
        if H[i+1,i] < tol
            warn("tol fail")
            break;
        end
        V[:,i+1]=w/H[i+1,i]
    end
    (Q,R) = qr(H,Val{false},thin=false)
    g = At_mul_B(Q,[β;zeros(numiters)])
    y = R \ g[1:numiters]
    x = V[:,1:numiters] * y
    x,g,V,H,Q,R,y
end

function givensGMRES(A,b;
                    numiters=10,
                    tol=sqrt(eps(eltype(A))))
    dim = LinAlg.checksquare(A)
    x = zeros(dim)
    β = norm(b)
    V = zeros(eltype(A),dim,numiters+1)
    V[:,1] = b/β
    H = zeros(eltype(A),numiters+1,numiters)
    for i = 1:numiters
        w=ortho!(view(H,:,i),
                 A*V[:,i],
                 (V[:,x] for x = 1:i))
        H[i+1,i] = norm(w)
        if H[i+1,i] < tol
            warn("tol fail")
            break;
        end
        V[:,i+1]=w/H[i+1,i]
    end
    R = copy(H)
    g = [β;zeros(numiters)]
    Q = LinAlg.Rotation{eltype(A)}([])
    for i = 1:numiters
        (Ω,h) = givens(view(R,:,i),i,i+1)
        R[i,i] = h
        R[i+1,i] = 0
        for j = i+1 : numiters
            A_mul_B!(Ω,view(R,:,j))
        end
        A_mul_B!(Ω,g)
        A_mul_B!(Ω,Q)
    end
    y = @view(R[1:numiters,:])\@view(g[1:numiters])
    x = @view(V[:,1:numiters])*y
    x,g,V,H,Q,R,y
end

function progressiveGMRES(A,b;
                          maxiters=10,
                          tol=sqrt(eps(eltype(A))))
    dim = LinAlg.checksquare(A)
    x = zeros(dim)
    β = norm(b)
    V = zeros(eltype(A),dim,maxiters+1)
    V[:,1] = b/β
    H = zeros(eltype(A),maxiters+1,maxiters)
    g = [β ; zeros(maxiters)]
    Q = LinAlg.Rotation{eltype(A)}([])
    R = zeros(eltype(A),maxiters,maxiters)
    for i = 1:maxiters
        w=ortho!(view(H,:,i),
                 A*V[:,i],
                 (V[:,x] for x = 1:i))
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
    @show(size(R),size(g),size(V))
    y = R\@view(g[1:end-1])
    x = @view(V[:,1:end-1]) * y
    x,g,V,H,Q,R,y
end


immutable CSPair{T}
    c::T
    s::T
end

cleanmat(A) = map(x -> abs(x) < sqrt(eps(eltype(A))) ? 0 : x, A)

end
