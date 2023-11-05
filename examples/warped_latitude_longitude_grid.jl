using OrthogonalSphericalShellGrids
using Oceananigans
using Oceananigans.Grids: generate_coordinate, R_Earth
using Oceananigans.Utils: get_cartesian_nodes_and_vertices
using GLMakie

grid = WarpedLatitudeLongitudeGrid(size = (360, 200, 1))
cartesian_nodes, cartesian_vertices = get_cartesian_nodes_and_vertices(grid, Center(), Center(), Center())

x, y, z = cartesian_nodes

fig = Figure()
ax  = LScene(fig[1, 1])

surface!(ax, x.*0.999, y.*0.999, z.*0.999, color = :blue)
wireframe!(ax, x, y, z)
