using Oceananigans.BoundaryConditions: fill_open_boundary_regions!, 
                                       permute_boundary_conditions, 
                                       fill_halo_event!,
                                       DistributedCommunication

using Oceananigans.DistributedComputations: cooperative_waitall!,
                                            recv_from_buffers!,
                                            fill_corners!,
                                            loc_id, 
                                            DCBCT

import Oceananigans.BoundaryConditions: fill_halo_regions!
import Oceananigans.DistributedComputations: synchronize_communication!

@inline instantiate(T::DataType) = T()
@inline instantiate(T) = T

const DistributedZipper = BoundaryCondition{<:DistributedCommunication, <:ZipperHaloCommunicationRanks}

switch_north_halos!(c, north_bc, grid, loc) = nothing

function switch_north_halos!(c, north_bc::DistributedZipper, grid, loc) 
    sign  = north_bc.condition.sign
    Hy = halo_size(grid)[2]
    Ny = size(grid)[2]
    sz = size(parent(c))

    _switch_north_halos!(parent(c), loc, sign, sz, Ny, Hy)

    return nothing
end

# We throw away the first point!
_switch_north_halos!(c, ::Tuple{<:Center, <:Center, <:Any}, sign, sz, Ny, Hy) = 
    view(c, :, Ny+Hy+1:Ny+2Hy-1, :) .= sign .* reverse(view(c, :, Ny+2Hy:-1:Ny+Hy+2, :), dims = 1) 

# We do not throw away the first point!
_switch_north_halos!(c, ::Tuple{<:Center, <:Face, <:Any}, sign, sz, Ny, Hy) = 
    view(c, :, Ny+Hy+1:Ny+2Hy, :) .= sign .* reverse(view(c, :, Ny+2Hy:-1:Ny+Hy+1, :), dims = 1) 

# We throw away the first line and the first point!
_switch_north_halos!(c, ::Tuple{<:Face, <:Center, <:Any}, sign, (Px, Py, Pz), Ny, Hy) = 
    view(c, 2:Px, Ny+Hy+1:Ny+2Hy-1, :) .= sign .* reverse(view(c, 2:Px, Ny+2Hy:-1:Ny+Hy+2, :), dims = 1)

# We throw away the first line but not the first point!
_switch_north_halos!(c, ::Tuple{<:Face, <:Face, <:Any}, sign, (Px, Py, Pz), Ny, Hy) = 
    view(c, 2:Px, Ny+Hy+1:Ny+2Hy, :) .= sign .* reverse(view(c, 2:Px, Ny+2Hy:-1:Ny+Hy+1, :), dims = 1)

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
    instantiated_location = map(instantiate, location(field))

    switch_north_halos!(field, north_bc, field.grid, instantiated_location)

    return nothing
end