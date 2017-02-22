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

immutable CSPair{T}
    c::T
    s::T
end

function optGMRES(A,b, x=zeros(b);
                  maxiters=100,
                  numvecs=10,
                  tol=sqrt(eps(eltype(A))))
    dim = size(b)[1]
    r₀ = b - A*x
    β = norm(r₀)
    V = CircularDeque{Vector{eltype(b)}}(numvecs)
    push!(V,r₀/β)
    h = zeros(eltype(b),numvecs+1)
    r = zeros(h)
    Q = CircularDeque{CSPair{eltype(b)}}(numvecs+1)
    γ = β
    P = CircularDeque{Vector{eltype(b)}}(numvecs)
    for i = 1:maxiters
        l = length(V)
        w=ortho!(h, A*back(V), V)
        h[l+1] = norm(w)
        for j = 1:l-1
            if j == 1
                r[1]=Q[j].s * h[1]
                r[2]=Q[j].c * h[2]
            else
                r[j] = Q[j].c*r[j] + Q[j].s*h[j]
                r[j+1] = -Q[j].s*r[j] + Q[j].c*h[j]
            end
        end
        (Ω,r[l]) = givens(r[l],h[l+1],1,2)
        Ωₙ = CSPair{eltype(b)}(Ω.c,Ω.s)
        length(Q) == numvecs+1 && shift!(Q)
        push!(Q, Ωₙ)
        γₙ = -Ωₙ.s*γ
        γ = Ωₙ.c*γ
        p = back(V)
        for j = 1:l-1
            LinAlg.axpy!(-r[j],P[j],p)
        end
        scale!(p,one(r[l])/r[l])
        length(P) == numvecs && shift!(P)
        push!(P,p)
        LinAlg.axpy!(γ,p,x)
        if abs(γₙ) < tol
            break
        end
        γ = γₙ
        if h[l+1] < tol
            error("ortho failure")
        end
        length(V) == numvecs && shift!(V)
        push!(V,w/h[l+1])
    end
    x
end

cleanmat(A) = map(x -> abs(x) < sqrt(eps(eltype(A))) ? 0 : x, A)

end
