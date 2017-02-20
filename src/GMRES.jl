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

end
