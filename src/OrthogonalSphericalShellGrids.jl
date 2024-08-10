module OrthogonalSphericalShellGrids

# The only things we need!
export TripolarGrid, ZipperBoundaryCondition

using Oceananigans
using Oceananigans: Face, Center
using Oceananigans.Grids: R_Earth

using Oceananigans.Fields: index_binary_search
using Oceananigans.Architectures: device, on_architecture
using JLD2
using Adapt 

using Oceananigans.Grids: halo_size, spherical_area_quadrilateral
using Oceananigans.Grids: lat_lon_to_cartesian
using Oceananigans.Utils: get_cartesian_nodes_and_vertices
using OffsetArrays
using Oceananigans.Operators

using KernelAbstractions: @kernel, @index
using KernelAbstractions.Extras.LoopInfo: @unroll
using Oceananigans.Grids: generate_coordinate

using Oceananigans.BoundaryConditions

@inline convert_to_0_360(x) = ((x % 360) + 360) % 360

using Oceananigans.Grids: topology
using Adapt

include("grid_utils.jl")
include("zipper_boundary_condition.jl")
include("generate_tripolar_coordinates.jl")
include("tripolar_grid.jl")
include("grid_extensions.jl")
include("distributed_tripolar_grid.jl")
include("with_halo.jl")
include("split_explicit_free_surface.jl")

end
