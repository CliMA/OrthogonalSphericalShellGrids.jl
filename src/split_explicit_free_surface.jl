using Oceananigans.Grids: halo_size, with_halo
using Oceananigans.Models.HydrostaticFreeSurfaceModels: SplitExplicitState,
                                                        SplitExplicitFreeSurface,
                                                        calculate_column_height!,
                                                        augmented_kernel_offsets,
                                                        augmented_kernel_size,
                                                        FixedTimeStepSize

import Oceananigans.Models.HydrostaticFreeSurfaceModels: materialize_free_surface, SplitExplicitAuxiliaryFields

# TODO: make sure this works on Distributed architectures, for the 
# moment the only distribution we allow with the tripolar grid is 
# a slab decomposition in the y-direction.
function SplitExplicitAuxiliaryFields(grid::TRG)

    Gᵁ = Field{Face,   Center, Nothing}(grid)
    Gⱽ = Field{Center, Face,   Nothing}(grid)

    # In a non-parallel grid we calculate only the interior,
    # otherwise, the rules remain the same
    kernel_size    = tripolar_augmented_kernel_size(grid)
    kernel_offsets = tripolar_augmented_kernel_offsets(grid)

    kernel_parameters = KernelParameters(kernel_size, kernel_offsets)
    
    return SplitExplicitAuxiliaryFields(Gᵁ, Gⱽ, kernel_parameters)
end

@inline tripolar_augmented_kernel_size(grid::TRG)    = (grid.Nx, grid.Ny + grid.Hy - 1)
@inline tripolar_augmented_kernel_offsets(grid::TRG) = (0, 0)
        
# In case of a distributed architecture, nothing changes!
@inline tripolar_augmented_kernel_size(grid::DTRG)    = augmented_kernel_size(grid)
@inline tripolar_augmented_kernel_offsets(grid::DTRG) = augmented_kernel_offsets(grid)

# We play the same trick as in the Distributed implementation and we extend the halos for
# a split explicit barotropic solver on a tripolar grid. Only on the North boundary though!
@inline tripolar_split_explicit_halos(old_halos, step_halo) = old_halos[1], max(step_halo, old_halos[2]), old_halos[3]

# Internal function for HydrostaticFreeSurfaceModel
function materialize_free_surface(free_surface::SplitExplicitFreeSurface, velocities, grid::TRG)

        settings  = free_surface.settings 

        if settings.substepping isa FixedTimeStepSize
                throw(ArgumentError("A variable substepping through a CFL condition is not supported for the `SplitExplicitFreeSurface` on a `TripolarGrid`. \n
                                     Provide a fixed number of substeps through the `substeps` keyword argument as: \n
                                     `free_surface = SplitExplicitFreeSurface(grid; substeps = N)` where `N::Int`"))
        end

        old_halos  = halo_size(grid)
        Nsubsteps  = length(settings.substepping.averaging_weights)

        extended_halos = tripolar_split_explicit_halos(old_halos, Nsubsteps+1)
        extended_grid  = with_halo(extended_halos, grid)

        Nze = size(extended_grid, 3)
        η = ZFaceField(extended_grid, indices = (:, :, Nze+1))

        return SplitExplicitFreeSurface(η,
                                        SplitExplicitState(extended_grid, settings.timestepper),
                                        SplitExplicitAuxiliaryFields(extended_grid),
                                        free_surface.gravitational_acceleration,
                                        free_surface.settings)
end
