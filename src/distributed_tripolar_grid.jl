using Oceananigans.DistributedComputations
using Oceananigans.DistributedComputations: ranks, inject_halo_communication_boundary_conditions

const DistributedTripolarGrid{FT, TX, TY, TZ, A, R, FR, Arch} = OrthogonalSphericalShellGrid{FT, TX, TY, TZ, A, R, FR, <:Tripolar, <:Distributed}

const DTRG = Union{DistributedTripolarGrid, ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:DistributedTripolarGrid}}

"""
    TripolarGrid(arch::Distributed, FT::DataType = Float64; halo = (4, 4, 4), kwargs...)

Constructs a tripolar grid on a distributed architecture.
The tripolar grid is supported only on a Y-partitioning configuration, 
therefore, only splitting the j-direction is supported for the moment.
"""
function TripolarGrid(arch::Distributed, FT::DataType = Float64; halo = (4, 4, 4), kwargs...)

    workers = ranks(arch.partition)

    workers[1] != 1 &&
        throw(ArgumentError("The tripolar grid is supported only on a Y-partitioning configuration"))
    
    Hx, Hy, Hz  = halo

    # We build the global grid on a CPU architecture, in order to split it easily
    global_grid = TripolarGrid(CPU(), FT; halo, kwargs...)
    Nx, Ny, Nz  = size(global_grid)

    # Splitting the grid manually, remember, only splitting 
    # the j-direction is supported for the moment
    lsize = local_size(arch, global_size)

    # Extracting the local range
    nlocal = concatenate_local_sizes(lsize, arch, 2)
    rank   = arch.local_rank
    jstart = 1 + sum(nlocal[1:rank-1])
    jend   = rank == workers[2] ? Ny : sum(nlocal[1:rank])
    jrange = jstart - Hy : jend + Hy

    # Partitioning the Coordinates
    λᶠᶠᵃ = grid.λᶠᶠᵃ[:, jrange]
    φᶠᶠᵃ = grid.φᶠᶠᵃ[:, jrange]
    λᶠᶜᵃ = grid.λᶠᶜᵃ[:, jrange]
    φᶠᶜᵃ = grid.φᶠᶜᵃ[:, jrange]
    λᶜᶠᵃ = grid.λᶜᶠᵃ[:, jrange]
    φᶜᶠᵃ = grid.φᶜᶠᵃ[:, jrange]
    λᶜᶜᵃ = grid.λᶜᶜᵃ[:, jrange]
    φᶜᶜᵃ = grid.φᶜᶜᵃ[:, jrange]

    # Partitioning the Metrics
    Δxᶜᶜᵃ = grid.Δxᶜᶜᵃ[:, jrange]
    Δxᶠᶜᵃ = grid.Δxᶠᶜᵃ[:, jrange]
    Δxᶜᶠᵃ = grid.Δxᶜᶠᵃ[:, jrange]
    Δxᶠᶠᵃ = grid.Δxᶠᶠᵃ[:, jrange]
    Δyᶜᶜᵃ = grid.Δyᶜᶜᵃ[:, jrange]
    Δyᶠᶜᵃ = grid.Δyᶠᶜᵃ[:, jrange]
    Δyᶜᶠᵃ = grid.Δyᶜᶠᵃ[:, jrange]
    Δyᶠᶠᵃ = grid.Δyᶠᶠᵃ[:, jrange]
    Azᶜᶜᵃ = grid.Azᶜᶜᵃ[:, jrange]
    Azᶠᶜᵃ = grid.Azᶠᶜᵃ[:, jrange]
    Azᶜᶠᵃ = grid.Azᶜᶠᵃ[:, jrange]
    Azᶠᶠᵃ = grid.Azᶠᶠᵃ[:, jrange]

    LY = rank == 0 ? RightConnected : FullyConnected 
    ny = nlocal[rank+1]

    zᵃᵃᶜ   = grid.zᵃᵃᶜ
    zᵃᵃᶠ   = grid.zᵃᵃᶠ
    Δzᵃᵃᶜ  = grid.Δzᵃᵃᶜ
    Δzᵃᵃᶠ  = grid.Δzᵃᵃᶠ
    radius = grid.radius

    grid = OrthogonalSphericalShellGrid{Periodic, LY, Bounded}(arch,
                    Nx, ny, Nz,
                    Hx, Hy, Hz,
                    convert(eltype(radius), Lz),
                    on_architecture(arch,  λᶜᶜᵃ), on_architecture(arch,  λᶠᶜᵃ), on_architecture(arch,  λᶜᶠᵃ), on_architecture(arch,  λᶠᶠᵃ),
                    on_architecture(arch,  φᶜᶜᵃ), on_architecture(arch,  φᶠᶜᵃ), on_architecture(arch,  φᶜᶠᵃ), on_architecture(arch,  φᶠᶠᵃ), on_architecture(arch, zᵃᵃᶜ),  on_architecture(arch, zᵃᵃᶠ),
                    on_architecture(arch, Δxᶜᶜᵃ), on_architecture(arch, Δxᶠᶜᵃ), on_architecture(arch, Δxᶜᶠᵃ), on_architecture(arch, Δxᶠᶠᵃ),
                    on_architecture(arch, Δyᶜᶜᵃ), on_architecture(arch, Δyᶜᶠᵃ), on_architecture(arch, Δyᶠᶜᵃ), on_architecture(arch, Δyᶠᶠᵃ), on_architecture(arch, Δzᵃᵃᶜ), on_architecture(arch, Δzᵃᵃᶠ),
                    on_architecture(arch, Azᶜᶜᵃ), on_architecture(arch, Azᶠᶜᵃ), on_architecture(arch, Azᶜᶠᵃ), on_architecture(arch, Azᶠᶠᵃ),
                    radius, Tripolar())
         
    return grid
end

#####
##### Boundary condition extensions
#####

# a distributed `TripolarGrid` needs a `ZipperBoundaryCondition` for the north boundary
# only on the last rank
function regularize_field_boundary_conditions(bcs::FieldBoundaryConditions,
                                              grid::DTRG,
                                              field_name::Symbol,
                                              prognostic_names=nothing)

    arch = architecture(grid)    
    loc  = assumed_field_location(field_name)
    rank = arch.local_rank
    workers = ranks(arch) 
    sign = field_name == :u || field_name == :v ? -1 : 1

    west   = regularize_boundary_condition(bcs.west,   grid, loc, 1, LeftBoundary,  prognostic_names)
    east   = regularize_boundary_condition(bcs.east,   grid, loc, 1, RightBoundary, prognostic_names)
    south  = regularize_boundary_condition(bcs.south,  grid, loc, 2, LeftBoundary,  prognostic_names)
    north  = if rank == workers[2] - 1
        ZipperBoundaryCondition(sign) 
    else
        regularize_boundary_condition(bcs.south, grid, loc, 2, RightBoundary, prognostic_names)
    end

    bottom = regularize_boundary_condition(bcs.bottom, grid, loc, 3, LeftBoundary,  prognostic_names)
    top    = regularize_boundary_condition(bcs.top,    grid, loc, 3, RightBoundary, prognostic_names)

    immersed = regularize_immersed_boundary_condition(bcs.immersed, grid, loc, field_name, prognostic_names)

    return FieldBoundaryConditions(west, east, south, north, bottom, top, immersed)
end

# Extension of the constructor for a `Field` on a `TRG` grid. We assumes that the north boundary is a zipper
# with a sign that depends on the location of the field (revert the value of the halos if on edges, keep it if on nodes or centers)
function Field((LX, LY, LZ)::Tuple, grid::DTRG, data, old_bcs, indices::Tuple, op, status)
    arch = architecture(grid)    
    rank = arch.local_rank
    workers = ranks(arch) 
    indices = validate_indices(indices, (LX, LY, LZ), grid)
    validate_field_data((LX, LY, LZ), data, grid, indices)
    validate_boundary_conditions((LX, LY, LZ), grid, old_bcs)
    default_zipper = ZipperBoundaryCondition(sign(LX, LY))

    new_bcs = inject_halo_communication_boundary_conditions(old_bcs, arch.local_rank, arch.connectivity, topology(grid))
    
    north_bc = if rank == workers[2] - 1 && !(new_bcs.north isa ZBC)
        default_zipper
    else
        new_bcs.north
    end
    
    new_bcs = FieldBoundaryConditions(; west  = new_bcs.west, 
                                        east  = new_bcs.east, 
                                        south = new_bcs.south,
                                        north = north_bc,
                                        top   = new_bcs.top,
                                        bottom = new_bcs.bottom)

    buffers = FieldBoundaryBuffers(grid, data, new_bcs)

    return Field{LX, LY, LZ}(grid, data, new_bcs, indices, op, status, buffers)
end
