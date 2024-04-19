using OrthogonalSphericalShellGrids

using Oceananigans
using Oceananigans.Units
using Printf
using OrthogonalSphericalShellGrids

underlying_grid = TripolarGrid(size = (360, 180, 1), halo = (5, 5, 5))

bottom_height = zeros(360, 180)

bottom_height[1:10, end-10:end]       .= 1
bottom_height[end-10:end, end-10:end] .= 1
bottom_height[170:190, end-10:end]   .= 1
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))

ψ(x, y, z) = - tanh((x .- 180) ./ 10) * 10

P = Field((Face, Face, Center), grid)
set!(P, ψ)

uI = compute!(Field(- ∂y(P)))
vI = compute!(Field(∂x(P)))

