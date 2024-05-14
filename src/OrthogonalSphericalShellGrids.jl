module OrthogonalSphericalShellGrids

export TripolarGrid, ZipperBoundaryCondition

using Oceananigans
using Oceananigans.Grids: R_Earth

using Oceananigans.Fields: index_binary_search
using Oceananigans.Architectures: device, on_architecture
using JLD2
using JLD2: @save, @load

using Oceananigans.Grids: halo_size, spherical_area_quadrilateral
using Oceananigans.Grids: lat_lon_to_cartesian
using OffsetArrays
using Oceananigans.Operators

using KernelAbstractions: @kernel, @index
using KernelAbstractions.Extras.LoopInfo: @unroll
using Oceananigans.Grids: generate_coordinate

using Oceananigans.BoundaryConditions
using CUDA

@inline convert_to_0_360(x) = ((x % 360) + 360) % 360

include("grid_utils.jl")
include("zipper_boundary_condition.jl")
include("generate_tripolar_coordinates.jl")
include("tripolar_grid.jl")
include("split_explicit_free_surface.jl")
include("with_halo.jl")

end
