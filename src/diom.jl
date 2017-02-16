
circoff(i, numvecs) = rem(i-1,numvecs)+1

function incomplete_ortho(V, A, iter, numvecs)
    orthotol = .1
    w = A*V[:, circoff(iter,numvecs)]
    initnorm = norm(w)
    fv = max(1, iter - numvecs + 1)
    h = zeros(eltype(V), iter-fv+2)
    for p =  fv : iter
        v = view(V,:,circoff(p,numvecs))
        h[p-fv+1] = dot(w, v)
        w -= h[p-fv+1] * v
    end
    finalnorm = norm(w)
    normratio = finalnorm / initnorm
    if normratio < orthotol
        warn("Orthoganalization cancellation failure, normratio $normratio, reorthogonalizing")
        for p = fv : iter
            v = view(V,:,circoff(p,numvecs))
            f = dot(w, v)
            w -= f * v
            h[p-fv+1] += f
        end        
    end
    h[iter-fv+2] = norm(w)
    w/h[iter-fv+2], h
end

function updateLU(l, u, h, ω, iter, numvecs)
    new_ω = last(h)
    fv = max(iter - numvecs + 1, 1)
    lv = min(iter,numvecs)
    if iter == 1
        u[1] = h[1]
        return new_ω
    end
    size(l,1) == numvecs-1 && shift!(l)
    push!(l, ω / u[min(iter-1,numvecs)])
    u[1] = h[1]
    for i = 2:size(h,1)-1
        u[i] = h[i] - h[i-1]*l[i-1]
    end
    return new_ω
end

function diom(A, b, numvecs=10)
    LinAlg.checksquare(A)
    x = zeros(eltype(A), size(A,2))
    β = norm(b)
    V = zeros(eltype(A), size(A,1), numvecs)
    V[:,1] = b/β
    ω = 0
    u = zeros(eltype(A), numvecs)
    l = zeros(eltype(A), 0)
    for iter = 1:5
        next_v, next_h = incomplete_ortho(V, A, iter, numvecs)
        @show(next_v)
        @show(next_h)
        V[:,circoff(iter+1,numvecs)] = next_v
        ω = updateLU(l, u, next_h, ω, iter, numvecs)
        @show(l)
        @show(u)
        @show(ω)
    end
    V
end
