
function gauss_seidel(A, b)::Vector{Float64}
    x_star = A\b
    width = size(A)[1]
    E = eye(width)
    x = zeros(width)
    err = Inf
    precision = 1.0e-6 * dot(b,b)
    iteration = 0
    while err > precision
        iteration += 1
        if iteration > 10000
            throw("Max iterations")
        end
        r = b - A*x
        dim=indmax(map(abs,r))
        #dim = mod(iteration-1,width)+1
        v = E[:,dim]
        err = dot(x-x_star,A*(x-x_star))
        ##imp = dot(r,v)^2 / A[dim,dim]
        println(iteration, " ", err, " ", r[dim], " ", dot(r,r))
        α = dot(v,r)/dot(v,A*v)
        x = x + α * v
    end
    return x
end
