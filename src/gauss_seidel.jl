
function gauss_seidel(A, b)::Vector{Float64}
    x_star = A\b
    width = size(A)[1]
    E = eye(width)
    x = zeros(width)
    err = 1.0
    while err > .00001
        r = b - A*x
        dim=indmax(map(abs,r))
        v = E[:,dim]
        err = dot(x-x_star,A*(x-x_star))
        ##imp = dot(r,v)^2 / A[dim,dim]
        println(err, " ", r[dim], " ", dot(r,r))
        α = dot(v,r)/dot(v,A*v)
        x = x + α * v
    end
    return x
end
