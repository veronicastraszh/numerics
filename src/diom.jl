
function incomplete_ortho!(h, V, A)
    orthotol = .1
    w = A*V[1]
    initnorm = norm(w)
    for (i,v) = enumerate(V)
        h[i] = dot(w, v)
        w -= h[i] * v
    end
    finalnorm = norm(w)
    normratio = finalnorm / initnorm
    if normratio < orthotol
        warn("Orthoganalization cancellation failure, normratio $normratio, reorthogonalizing")
        for (i,v) = enumerate(V)
            f = dot(w, v)
            @show(f)
            w -= f * v
            h[i] += f
        end        
    end
    w
end

function updateLU!(u,l,h,nl)
    for i = size(l,1):-1:2
        l[i]=l[i-1]
    end
    l[1] = nl
    u[end] = h[end]
    for i in size(h,1)-1:-1:1
        u[i]=h[i]-u[i+1]*l[i]
    end
    nothing
end

function diom(A, b, numvecs=10; tol=sqrt(eps(eltype(A))))
    LinAlg.checksquare(A)
    x = zeros(eltype(A), size(b,1))
    β = norm(b)
    V = Any[b/β]
    ω_saved = 0
    h = zeros(eltype(A), numvecs)
    u = zeros(eltype(A), numvecs)
    u[1] = 1
    l = zeros(eltype(A), numvecs-1)
    local ω
    for iter = 1:5
        w = incomplete_ortho!(h, V, A)
        ω = norm(w)
        @show(ω)
        @show(h)
        if ω >= tol
            unshift!(V,w/ω)
            (size(V,1) > numvecs) && pop!(V)
            updateLU!(u,l,h,ω_saved/u[1])
            ω_saved = ω
            @show(l)
            @show(u)
        else
        end
    end
    (ω,h,l,u)
end

function checkV(V)
    for (i,v) = enumerate(V)
        for j = i:size(V,1)
            println("($i,$j) $(dot(V[i],V[j]))")
        end
    end
end
