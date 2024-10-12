using OrthogonalSphericalShellGrids
using Oceananigans
using Oceananigans.Utils
using Oceananigans.Grids: halo_size, φnodes, λnodes
using Oceananigans.Utils
using Oceananigans.BoundaryConditions
using OrthogonalSphericalShellGrids: get_cartesian_nodes_and_vertices
using Oceananigans.CUDA
using Test

using KernelAbstractions: @kernel, @index

arch = CUDA.has_cuda_gpu() ? GPU() : CPU()

# Mask the singularity of the grid in a region of 
# 5 degrees radius around the singularities
function mask_singularities(underlying_grid::TripolarGrid)
    λp = underlying_grid.conformal_mapping.first_pole_longitude
    φp = underlying_grid.conformal_mapping.north_poles_latitude
    
    # We need a bottom height field that ``masks'' the singularities
    bottom_height(λ, φ) = ((abs(λ - λp) < 5)       & (abs(φp - φ) < 5)) |
                          ((abs(λ - λp - 180) < 5) & (abs(φp - φ) < 5)) | (φ < -80) ? 0 : - 1000

    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))

    return grid
end
