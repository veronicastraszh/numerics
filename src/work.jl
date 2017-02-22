
import GMRES

A = sprandn(500,500,0.01)

A += speye(500) * 5

b = randn(500)

nothing
