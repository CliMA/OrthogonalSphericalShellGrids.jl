module OrthogonalSphericalShellGrids

# The only things we need!
export TripolarGrid, ZipperBoundaryCondition

using Oceananigans
using Oceananigans: Face, Center
using Oceananigans.Architectures: device, on_architecture
using Oceananigans.BoundaryConditions
using Oceananigans.Fields: index_binary_search
using Oceananigans.Grids: R_Earth, 
                          halo_size, spherical_area_quadrilateral,
                          lat_lon_to_cartesian, generate_coordinate, topology
using Oceananigans.Operators
using Oceananigans.Utils: get_cartesian_nodes_and_vertices

using Adapt 
using KernelAbstractions: @kernel, @index
using KernelAbstractions.Extras.LoopInfo: @unroll
using OffsetArrays

@inline convert_to_0_360(x) = ((x % 360) + 360) % 360

include("tripolar_grid_utils.jl")
include("zipper_boundary_condition.jl")
include("generate_tripolar_coordinates.jl")
include("tripolar_grid.jl")
include("tripolar_grid_extensions.jl")
include("distributed_tripolar_grid.jl")
include("with_halo.jl")
include("split_explicit_free_surface.jl")

end
