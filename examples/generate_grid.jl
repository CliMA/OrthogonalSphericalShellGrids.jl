using OrthogonalSphericalShellGrids
using OrthogonalSphericalShellGrids: get_cartesian_nodes_and_vertices
using CairoMakie

# Generate a Tripolar grid with a 2 degree resolution and ``north'' singularities at 20 degrees latitude
grid = OrthogonalSphericalShellGrids.TripolarGrid(size = (180, 90, 1), north_poles_latitude = 35)

# retrieve the Face-Face nodes in a Cartesian coordinate system
cartesian_nodes, _ = get_cartesian_nodes_and_vertices(grid, Face(), Face(), Center())
xF, yF, zF = cartesian_nodes

# retrieve the Center-Center nodes in a Cartesian coordinate system
cartesian_nodes, _ = get_cartesian_nodes_and_vertices(grid, Center(), Center(), Center())
xC, yC, zC = cartesian_nodes

# Plot a nice visualization of the grid structure
fig = Figure()
ax  = LScene(fig[1, 1]; show_axis = false)

surface!(ax, xF.*0.9999, yF.*0.9999, zF.*0.9999, color = :blue)
wireframe!(ax, xF, yF, zF, color = :black)

surface!(ax, xC.*0.9999, yC.*0.9999, zC.*0.9999, color = :blue)
wireframe!(ax, xC, yC, zC, color = :blue)

fig
