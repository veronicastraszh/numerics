
circoff(i, numvecs) = rem(i-1,numvecs)+1

function incomplete_ortho(V, A, iter, numvecs)
    orthotol = .1
    w = A*V[:, circoff(iter,numvecs)]
    initnorm = norm(w)
    h = zeros(eltype(V), iter + 1)
    for p = max(1, iter - numvecs + 1) : iter
        v = view(V,:,circoff(p,numvecs))
        h[p] = dot(w, v)
        w -= h[p] * v
    end
    finalnorm = norm(w)
    normratio = finalnorm / initnorm
    if normratio < orthotol
        warn("Orthoganalization cancellation failure, normratio $normratio, reorthogonalizing")
        for p = max(1, iter - numvecs + 1) : iter
            v = view(V,:,circoff(p,numvecs))
            f = dot(w, v)
            w -= f * v
            h[p] += f
        end        
    end
    h[iter+1] = norm(w)
    w/h[iter+1], h
end

function diom(A, b, numvecs=5)
    x = zeros(eltype(A), size(A,2))
    β = norm(b)
    V = zeros(eltype(A), size(A,1), numvecs)
    H = zeros(eltype(A), numvecs+1, numvecs)
    V[:,1] = b/β
    for iter = 1:3
        next_v, next_h = incomplete_ortho(V, A, iter, numvecs)
        @show(next_v)
        @show(next_h)
        V[:,circoff(iter+1,numvecs)] = next_v
        htop = max(1, iter - numvecs + 1)
        hbot = htop + size(next_h, 1) - 1
        H[htop:hbot ,circoff(iter,numvecs)] = next_h
    end
    V, H
end
