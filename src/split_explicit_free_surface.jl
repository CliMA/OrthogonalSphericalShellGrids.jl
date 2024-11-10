using Oceananigans.Grids: halo_size, with_halo

using Oceananigans.Models.HydrostaticFreeSurfaceModels: SplitExplicitState,
                                                        SplitExplicitFreeSurface,
                                                        augmented_kernel_offsets,
                                                        augmented_kernel_size,
                                                        FixedTimeStepSize

import Oceananigans.Models.HydrostaticFreeSurfaceModels: materialize_free_surface, SplitExplicitAuxiliaryFields

positive_zipper_boundary(default_field, ::TRG) =
    FieldBoundaryConditions(top = nothing,
                            bottom = nothing,
                            west = default_field.boundary_conditions.west,
                            east = default_field.boundary_conditions.east,
                            south = default_field.boundary_conditions.south,
                            north = ZipperBoundaryCondition())

function positive_zipper_boundary(default_field, grid::DTRG)  
        arch = architecture(grid)
        workers = ranks(arch.partition)

        if arch.local_rank == workers[2] - 1
            return FieldBoundaryConditions(top = nothing,
                                           bottom = nothing,
                                           west = default_field.boundary_conditions.west,
                                           east = default_field.boundary_conditions.east,
                                           south = default_field.boundary_conditions.south,
                                           north = ZipperBoundaryCondition())
        else
            return default_field.boundary_conditions
        end
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
