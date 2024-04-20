using Oceananigans.Grids: architecture, cpu_face_constructor_z

import Oceananigans.Grids: with_halo

function with_halo(new_halo, old_grid::TripolarGrid)

    size = (old_grid.Nx, old_grid.Ny, old_grid.Nz)

    z = cpu_face_constructor_z(old_grid)

    new_grid = TripolarGrid(architecture(old_grid), eltype(old_grid);
                                            size, z, 
                                            radius = old_grid.radius,
                                            halo = new_halo)

    return new_grid
end