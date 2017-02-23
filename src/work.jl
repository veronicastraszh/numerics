
push!(LOAD_PATH, pwd())

import GMRES

A = sprandn(500,500,0.01)
scale!(A,.1)
A += speye(500) * 2

b = randn(500)

nothing
