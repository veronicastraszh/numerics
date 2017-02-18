
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

function computeP(P,u,v)
    new_p = copy(v)
    for (i,p) = enumerate(P)
        new_p -= p*u[i+1]
    end
    return new_p / u[1]
end

function diom(A, b;
              numvecs=10,
              tol=sqrt(eps(eltype(A))),
              numiters=100,
              )
    dim = LinAlg.checksquare(A)
    x = zeros(eltype(A), dim)
    β = norm(b)
    V = [b/β]
    P = Array{Vector{eltype(A)}}(0)
    ω_saved = 0
    h = zeros(eltype(A), numvecs)
    u = zeros(eltype(A), numvecs)
    u[1] = 1
    l = zeros(eltype(A), numvecs-1)
    local ω, ζ
    for iter = 1:numiters
        w = incomplete_ortho!(h, V, A)
        ω = norm(w)
        if abs(ω) >= tol
            new_v = w/ω
            updateLU!(u,l,h,ω_saved/u[1])
            ω_saved = ω
            @show(h,ω,u,l)
            if abs(u[1]) >= tol
                new_p = computeP(P,u,V[1])
                ζ = (iter == 1) ? β : -l[1]*ζ
                @show(ζ)
                x += ζ*new_p
                unshift!(V,new_v)
                length(V) > numvecs && pop!(V)
                unshift!(P,new_p)
                length(P) >= numvecs && pop!(P)
            else
                error("u[1] below tol")
            end
        else
            warn("ω below tol")
            return x
        end
        (@show(ω*abs(ζ/u[1])) < tol) && break
    end
    x
end

function fiom(A, b;
              numiters=10,
              tol=sqrt(eps(eltype(A))),
              )
    dim = LinAlg.checksquare(A)
    H = zeros(eltype(A),numiters+1,numiters)
    β = norm(b)
    VA = [b/β]
    for iter = 1:numiters
        w = incomplete_ortho!(view(H,iter:-1:1,iter),
                              VA,
                              A)
        ω = norm(w)
        if (abs(ω) > tol)
            H[iter+1,iter] = ω
            unshift!(VA,w/ω)
        else
            error("ω tolerance")
        end
    end
    V = foldl(hcat,VA[end:-1:1])
    lu = lufact(H[1:numiters,:],Val{false})
    y = lu \ vcat(β,zeros(numiters-1))
    V,H,lu[:L],lu[:U],y,V[:,1:numiters]*y
end


function checkV(V)
    for (i,v) = enumerate(V)
        for j = i:size(V,1)
            println("($i,$j) $(dot(V[i],V[j]))")
        end
    end
end
