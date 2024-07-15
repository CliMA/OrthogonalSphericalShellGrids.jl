# # Generate a Tripolar Grid
#
# This Examples shows how to generate and visualize a Tripolar grid using the `OrthogonalSphericalShellGrids`
# and CairoMakie packages. The script is structured into several key sections, each performing a specific task in the process
# of creating and displaying the grid.
#
# The script begins by importing the required Julia packages. 
# OrthogonalSphericalShellGrids is used for generating the Tripolar grid, a type of grid used in geophysical and oceanographic modeling 
# due to its ability to handle polar singularities effectively. 
# CairoMakie is used for visualization purposes, allowing the creation of figures and plots to display the grid.
#

using OrthogonalSphericalShellGrids
using OrthogonalSphericalShellGrids: Face, Center
using OrthogonalSphericalShellGrids: get_cartesian_nodes_and_vertices
using CairoMakie

# ## Grid generation
#
# The grid is generated with a call to OrthogonalSphericalShellGrids.TripolarGrid, 
# specifying the grid size and the latitude of the "north" singularities. In this case, the grid has a resolution of 2 degrees 
# (180x90 grid points) and north singularities at 35 degrees latitude. This setup creates a grid that covers the globe with a 
# specific focus on handling the complexities near the poles.

grid = OrthogonalSphericalShellGrids.TripolarGrid(size = (180, 90, 1), north_poles_latitude = 35)

# ## Retriveing the nodes
#
# The lines retrieve Cartesian coordinates for two types of grid nodes: `Face`-`Face` nodes 
# and `Center`-`Center`` nodes. This is done using the `get_cartesian_nodes_and_vertices` function, 
# which converts the grid's spherical coordinates into Cartesian coordinates for easier manipulation and visualization. 
# The coordinates are stored in variables xF, yF, zF for Face-Face nodes, and xC, yC, zC for Center-Center nodes.

cartesian_nodes, _ = get_cartesian_nodes_and_vertices(grid, Face(), Face(), Center())
xF, yF, zF = cartesian_nodes

cartesian_nodes, _ = get_cartesian_nodes_and_vertices(grid, Center(), Center(), Center())
xC, yC, zC = cartesian_nodes

# ## Visualization
# 
# Finally, we can visualize the grid using two types of visual elements: surfaces and wireframes.
# Surfaces are plotted slightly smaller (0.9999 scale) than the wireframes to create a layered effect,
# enhancing the visual distinction between the grid's structure and its outline. 
# The Face-Face nodes are visualized with blue surfaces and black wireframes, 
# while the Center-Center nodes are visualized with blue surfaces and wireframes. 

fig = Figure()
ax  = LScene(fig[1, 1]; show_axis = false)

surface!(ax, xF.*0.9999, yF.*0.9999, zF.*0.9999, color = :blue)
wireframe!(ax, xF, yF, zF, color = :black)

surface!(ax, xC.*0.9999, yC.*0.9999, zC.*0.9999, color = :blue)
wireframe!(ax, xC, yC, zC, color = :blue)

save("tripolar_grid_nodes.png", fig)
nothing #hide

# ![](tripolar_grid_nodes.png)


