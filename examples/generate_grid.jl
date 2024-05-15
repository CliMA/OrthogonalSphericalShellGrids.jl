using Oceananigans
using Oceananigans.Grids: R_Earth, generate_coordinate
using Oceananigans.Utils: get_cartesian_nodes_and_vertices
using OrthogonalSphericalShellGrids
using GLMakie

grid = OrthogonalSphericalShellGrids.TripolarGrid(size = (90, 45, 1), north_poles_latitude = 0)
cartesian_nodes, cartesian_vertices = get_cartesian_nodes_and_vertices(grid, Face(), Face(), Center())
xF, yF, zF = cartesian_nodes

cartesian_nodes, cartesian_vertices = get_cartesian_nodes_and_vertices(grid, Center(), Center(), Center())
xC, yC, zC = cartesian_nodes

fig = Figure()
ax  = LScene(fig[1, 1])

surface!(ax, xF.*0.9999, yF.*0.9999, zF.*0.9999, color = :blue)
wireframe!(ax, xF, yF, zF, color = :black)

surface!(ax, xC.*0.9999, yC.*0.9999, zC.*0.9999, color = :blue)
wireframe!(ax, xC, yC, zC, color = :blue)
