# # Generate a Tripolar Grid
#
# This example demonstrates how to generate a tripolar grid using the OrthogonalSphericalShellGrids
# package and how to visualize the grid using the CairoMakie package. The example is structured
# into several key sections, each performing a specific task in the process of creating and
# visualizing the grid.
#
# We begin by importing the required Julia packages.
# OrthogonalSphericalShellGrids is used for generating the tripolar grid, a type of grid used in
# geophysical and oceanographic modeling due to its ability to handle polar singularities effectively.
# CairoMakie is used for visualization purposes, allowing the creation of figures and plots to visualize
# the grid.

using OrthogonalSphericalShellGrids
using OrthogonalSphericalShellGrids: Face, Center
using OrthogonalSphericalShellGrids: get_cartesian_nodes_and_vertices
using GLMakie

# ## Grid generation
#
# The grid is generated via `OrthogonalSphericalShellGrids.TripolarGrid` method by
# specifying the grid size and the latitude of the "north" singularities. Here, we use
# a lateral resolution of 6 degrees (60 x 30 grid points in the horizontal) and we
# configure the North hemisphere singularities to be at 60 degrees latitude. This setup
# creates a grid that covers the globe with a specific focus on handling the complexities
# near the North pole.

grid = OrthogonalSphericalShellGrids.TripolarGrid(size = (60, 30, 1),
                                                  north_poles_latitude = 60)

# ## Retrieving the nodes
#
# The lines retrieve Cartesian coordinates for two types of grid nodes: `Face`--`Face` nodes 
# and `Center`--`Center`` nodes. This is done using the `get_cartesian_nodes_and_vertices` method,
# which converts the grid's spherical coordinates into Cartesian coordinates for easier manipulation
# and visualization. The coordinates are stored in variables `xF`, `yF`, `zF` for Face-Face nodes,
# and `xC`, `yC`, `zC` for Center-Center nodes.

cartesian_nodes, _ = get_cartesian_nodes_and_vertices(grid, Face(), Face(), Center())
xF, yF, zF = cartesian_nodes

cartesian_nodes, _ = get_cartesian_nodes_and_vertices(grid, Center(), Center(), Center())
xC, yC, zC = cartesian_nodes

# ## Visualization
# 
# Finally, we can visualize the grid using two types of visual elements: surfaces and wireframes.
# Surfaces are plotted slightly smaller (at a scale of 0.99 instead of 1) than the wireframes
# to create a layered effect, enhancing the visual distinction between the grid's structure and
# its outline. The Face-Face nodes are visualized with blue surfaces and black wireframes,
# while the Center-Center nodes are visualized with blue surfaces and wireframes.

fig = Figure(size=(1200, 600))
axN = Axis3(fig[1, 1]; aspect=(1, 1, 1), elevation = +0.9, azimuth = 7, height=1500)
axS = Axis3(fig[1, 2]; aspect=(1, 1, 1), elevation = -0.9, azimuth = 7, height=1500)

for ax in (axN, axS)
    scale_factor = 0.99
    
    surface!(ax, xF .* scale_factor, yF .* scale_factor, zF .* scale_factor, color = :blue)
    wireframe!(ax, xF, yF, zF, color = :black)

    surface!(ax, xC .* scale_factor, yC .* scale_factor, zC .* scale_factor, color = :blue)
    wireframe!(ax, xC, yC, zC, color = :blue)

    scatter!(ax, 0, 0, +1, color=:red, markersize=15)
    scatter!(ax, 0, 0, -1, color=:red, markersize=15)

    hidedecorations!(ax)
    hidespines!(ax)
end

save("tripolar_grid_nodes.png", fig)
nothing #hide

# ![](tripolar_grid_nodes.png)

# On the left we see how the North hemisphere has two poles that are shifted at latitude 60; on
# the right we see that the Southern hemisphere has its pole at the usual South Pole.
#
# This tripolar configuration are very typical for ocean modeling so that the pole singularity in
# the Arctic ocean is moved away into the land mases of Russia and Canada; the South pole
# singularity is already within the land mass of Antarctica so it's all good down there!
