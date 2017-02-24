
push!(LOAD_PATH, pwd())

import GMRES

dim=100000
A = sprandn(dim,dim,0.001)
scale!(A,.1)
A += speye(dim)

b = randn(dim)

nothing
