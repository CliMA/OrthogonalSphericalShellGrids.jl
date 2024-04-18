using Oceananigans
using Oceananigans.Grids: R_Earth, generate_coordinate
using Oceananigans.Utils: get_cartesian_nodes_and_vertices
using OrthogonalSphericalShellGrids
using GLMakie

grid = OrthogonalSphericalShellGrids.TripolarGrid(size = (720, 300, 1))
cartesian_nodes, cartesian_vertices = get_cartesian_nodes_and_vertices(grid, Face(), Face(), Center())

x, y, z = cartesian_nodes

fig = Figure()
ax  = LScene(fig[1, 1])

surface!(ax, x.*0.9999, y.*0.9999, z.*0.9999, color = :blue)
wireframe!(ax, x[1:end, 1:end], y[1:end, 1:end], z[1:end, 1:end])
