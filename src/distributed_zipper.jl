using Oceananigans.BoundaryConditions: fill_open_boundary_regions!, 
                                       permute_boundary_conditions, 
                                       fill_halo_event!,
                                       fill_corners!,
                                       DistributedCommunication

using Oceananigans.DistributedComputations: cooperative_waitall!,
                                            recv_from_buffers!,
                                            loc_id

import Oceananigans.BoundaryConditions: fill_halo_regions!
import Oceananigans.DistributedComputations: synchronize_communication
import Oceananigans.DistributedComputations: north_recv_tag, north_send_tag

const DistributedZipper = BoundaryCondition{<:DistributedCommunication, <:ZipperHaloCommunicationRanks}

ID_DIGITS = 2

sides  = (:west, :east, :south, :north, :southwest, :southeast, :northwest, :northeast)
side_id = Dict(side => n-1 for (n, side) in enumerate(sides))

# Change these and we are golden!
function north_recv_tag(arch, grid::DTRG, location)
    field_id   = string(arch.mpi_tag[], pad=ID_DIGITS)
    loc_digit  = string(loc_id(location...), pad=ID_DIGITS)
    last_rank  = arch.local_index[2] == ranks(arch)[2]
    side_digit = last_rank ? 9 : string(side_id[:south])
    return parse(Int, field_id * loc_digit * side_digit)
end

function north_send_tag(arch, grid::DTRG, location)
    field_id   = string(arch.mpi_tag[], pad=ID_DIGITS)
    loc_digit  = string(loc_id(location...), pad=ID_DIGITS)
    last_rank  = arch.local_index[2] == ranks(arch)[2]
    side_digit = last_rank ? 9 : string(side_id[:north])
    return parse(Int, field_id * loc_digit * side_digit)
end

function northwest_recv_tag(arch, grid::DTRG, location)
    field_id   = string(arch.mpi_tag[], pad=ID_DIGITS)
    loc_digit  = string(loc_id(location...), pad=ID_DIGITS)
    last_rank  = arch.local_index[2] == ranks(arch)[2]
    side_digit = last_rank ? 9 : string(side_id[:southeast])
    return parse(Int, field_id * loc_digit * side_digit)
end

function northwest_send_tag(arch, grid::DTRG, location)
    field_id   = string(arch.mpi_tag[], pad=ID_DIGITS)
    loc_digit  = string(loc_id(location...), pad=ID_DIGITS)
    last_rank  = arch.local_index[2] == ranks(arch)[2]
    side_digit = last_rank ? 9 : string(side_id[:northwest])
    return parse(Int, field_id * loc_digit * side_digit)
end

function northeast_recv_tag(arch, grid::DTRG, location)
    field_id   = string(arch.mpi_tag[], pad=ID_DIGITS)
    loc_digit  = string(loc_id(location...), pad=ID_DIGITS)
    last_rank  = arch.local_index[2] == ranks(arch)[2]
    side_digit = last_rank ? 9 : string(side_id[:southwest])
    return parse(Int, field_id * loc_digit * side_digit)
end

function northeast_send_tag(arch, grid::DTRG, location)
    field_id   = string(arch.mpi_tag[], pad=ID_DIGITS)
    loc_digit  = string(loc_id(location...), pad=ID_DIGITS)
    last_rank  = arch.local_index[2] == ranks(arch)[2]
    side_digit = last_rank ? 9 : string(side_id[:northeast])
    return parse(Int, field_id * loc_digit * side_digit)
end

switch_north_halos!(c, north_bc, grid, loc) = nothing

function switch_north_halos!(c, north_bc::DistributedZipper, grid, loc) 
    sign = north_bc.condition.sign

    params = ...

    launch!(architecture(grid), grid, params, grid, loc, sign, c)

    return nothing
end

@kernel function _switch_north_halos!(i, k, grid, ::Tuple{<:Face, <:Face, <:Any}, sign, c)
    Nx, Ny, _ = size(grid)
    
    i′ = Nx - i + 2 # Remember! element Nx + 1 does not exist!
    s  = ifelse(i′ > Nx , abs(sign), sign) # for periodic elements we change the sign
    i′ = ifelse(i′ > Nx, i′ - Nx, i′) # Periodicity is hardcoded in the x-direction!!
    Hy = grid.Hy
    
    for j = 1 : Hy
        @inbounds begin
            c[i, Ny + j, k] = s * c[i′, Ny - j + 1, k] 
        end
    end

    return nothing
end

@kernel function _switch_north_halos!(i, k, grid, ::Tuple{<:Face, <:Center, <:Any}, sign, c)
    Nx, Ny, _ = size(grid)
    
    i′ = Nx - i + 2 # Remember! element Nx + 1 does not exist!
    s  = ifelse(i′ > Nx , abs(sign), sign) # for periodic elements we change the sign
    i′ = ifelse(i′ > Nx, i′ - Nx, i′) # Periodicity is hardcoded in the x-direction!!
    Hy = grid.Hy
    
    for j = 1 : Hy
        @inbounds begin
            c[i, Ny + j, k] = s * c[i′, Ny - j, k] # The Ny line is duplicated so we substitute starting Ny-1
        end
    end

    return nothing
end

@kernel function _switch_north_halos!(i, k, grid, ::Tuple{<:Center, <:Face, <:Any}, sign, c)
    Nx, Ny, _ = size(grid)
    
    i′ = Nx - i + 1
    Hy = grid.Hy
    
    for j = 1 : Hy
        @inbounds begin
            c[i, Ny + j, k] = sign * c[i′, Ny - j + 1, k] 
        end
    end

    return nothing
end

@kernel function _switch_north_halos!(i, k, grid, ::Tuple{<:Center, <:Center, <:Any}, sign, c)
    Nx, Ny, _ = size(grid)
    
    i′ = Nx - i + 1
    Hy = grid.Hy
    
    for j = 1 : Hy
        @inbounds begin
            c[i, Ny + j, k] = sign * c[i′, Ny - j, k] # The Ny line is duplicated so we substitute starting Ny-1
        end
    end

    return nothing
end

function fill_halo_regions!(c::OffsetArray, bcs, indices, loc, grid::DTRG, buffers, args...; fill_boundary_normal_velocities = true, kwargs...)
    if fill_boundary_normal_velocities
        fill_open_boundary_regions!(c, bcs, indices, loc, grid, args...; kwargs...)
    end
    
    north_bc = bcs.north

    arch = architecture(grid)
    fill_halos!, bcs = permute_boundary_conditions(bcs) 

    number_of_tasks = length(fill_halos!)

    for task = 1:number_of_tasks
        fill_halo_event!(c, fill_halos![task], bcs[task], indices, loc, arch, grid, buffers, args...; kwargs...)
    end

    fill_corners!(c, arch.connectivity, indices, loc, arch, grid, buffers, args...; kwargs...)
    
    # We increment the tag counter only if we have actually initiated the MPI communication.
    # This is the case only if at least one of the boundary conditions is a distributed communication 
    # boundary condition (DCBCT) _and_ the `only_local_halos` keyword argument is false.
    increment_tag = any(isa.(bcs, DCBCT)) && !only_local_halos
    
    if increment_tag 
        arch.mpi_tag[] += 1
    end
        
    switch_north_halos!(parent(c), north_bc, grid, loc)
    
    return nothing
end

function synchronize_communication!(field)
    arch = architecture(field.grid)

    # Wait for outstanding requests
    if !isempty(arch.mpi_requests) 
        cooperative_waitall!(arch.mpi_requests)

        # Reset MPI tag
        arch.mpi_tag[] = 0

        # Reset MPI requests
        empty!(arch.mpi_requests)
    end

    recv_from_buffers!(field.data, field.boundary_buffers, field.grid)

    north_bc = field.boundary_conditions.north
    switch_north_halos!(parent(field.data), north_bc, field.grid, location(field))

    return nothing
end