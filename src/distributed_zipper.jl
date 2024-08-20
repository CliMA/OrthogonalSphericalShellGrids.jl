using Oceananigans.BoundaryConditions: fill_open_boundary_regions!, 
                                       permute_boundary_conditions, 
                                       fill_halo_event!,
                                       DistributedCommunication

using Oceananigans.DistributedComputations: cooperative_waitall!,
                                            recv_from_buffers!,
                                            fill_corners!,
                                            loc_id, 
                                            DCBCT

using Oceananigans.Utils: KernelParameters

import Oceananigans.BoundaryConditions: fill_halo_regions!
import Oceananigans.DistributedComputations: synchronize_communication!

import Oceananigans.Fields: create_buffer_y, create_buffer_corner

const DistributedZipper = BoundaryCondition{<:DistributedCommunication, <:ZipperHaloCommunicationRanks}

switch_north_halos!(c, north_bc, grid, loc) = nothing

function switch_north_halos!(c, north_bc::DistributedZipper, grid, loc) 
    sign   = north_bc.condition.sign
    Hx, Hy, _  = halo_size(grid)
    Nx, Ny, Nz = size(grid)

    params = KernelParameters((Nx+2Hx-2, Nz), (-Hx+1, 0))

    launch!(architecture(grid), grid, params, _switch_north_halos!, grid, loc, sign, c)

    return nothing
end

@kernel function _switch_north_halos!(grid, ::Tuple{<:Face, <:Center, <:Any}, sign, c)
    i, k = @index(Global, NTuple)
    Nx, Ny, _ = size(grid)
    
    i′ = Nx - i + 2 
    Hy = grid.Hy
    
    for j = 1 : Hy - 1 # TO CORRECTED!!!
        @inbounds c[i, Ny + j, k] = sign * c[i′, Ny + Hy - j, k] 
    end
end

@kernel function _switch_north_halos!(grid, ::Tuple{<:Center, <:Face, <:Any}, sign, c)
    i, k = @index(Global, NTuple)
    Nx, Ny, _ = size(grid)
    
    i′ = Nx - i + 1
    Hy = grid.Hy
    
    for j = 1 : Hy - 1
        @inbounds c[i, Ny + j, k] = sign * c[i′, Ny + Hy - j + 1, k] 
    end
end

@kernel function _switch_north_halos!(grid, ::Tuple{<:Center, <:Center, <:Any}, sign, c)
    i, k = @index(Global, NTuple)
    Nx, Ny, _ = size(grid)
    
    i′ = Nx - i + 1
    Hy = grid.Hy
    
    for j = 1 : Hy - 1
        @inbounds c[i, Ny + j, k] = sign * c[i′, Ny + Hy - j, k]
    end
end

function fill_halo_regions!(c::OffsetArray, bcs, indices, loc, grid::DTRG, buffers, args...; only_local_halos = false, fill_boundary_normal_velocities = true, kwargs...)
    if fill_boundary_normal_velocities
        fill_open_boundary_regions!(c, bcs, indices, loc, grid, args...; kwargs...)
    end
    
    north_bc = bcs.north

    arch = architecture(grid)
    fill_halos!, bcs = permute_boundary_conditions(bcs) 

    number_of_tasks = length(fill_halos!)

    for task = 1:number_of_tasks
        fill_halo_event!(c, fill_halos![task], bcs[task], indices, loc, arch, grid, buffers, args...; only_local_halos, kwargs...)
    end

    fill_corners!(c, arch.connectivity, indices, loc, arch, grid, buffers, args...; only_local_halos, kwargs...)
    
    # We increment the tag counter only if we have actually initiated the MPI communication.
    # This is the case only if at least one of the boundary conditions is a distributed communication 
    # boundary condition (DCBCT) _and_ the `only_local_halos` keyword argument is false.
    increment_tag = any(isa.(bcs, DCBCT)) && !only_local_halos
    
    if increment_tag 
        arch.mpi_tag[] += 1
    end
        
    switch_north_halos!(c, north_bc, grid, loc)
    
    return nothing
end

function synchronize_communication!(field::Field{<:Any, <:Any, <:Any, <:Any, <:DTRG})
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
    switch_north_halos!(field, north_bc, field.grid, location(field))

    return nothing
end