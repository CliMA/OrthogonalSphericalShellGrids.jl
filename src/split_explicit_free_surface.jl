using Oceananigans.Grids: halo_size, with_halo
using Oceananigans.Models.HydrostaticFreeSurfaceModels: SplitExplicitState, SplitExplicitFreeSurface, calculate_column_height! 

import Oceananigans.Models.HydrostaticFreeSurfaceModels: materialize_free_surface, SplitExplicitAuxiliaryFields
import Oceananigans.Models.HydrostaticFreeSurfaceModels: augmented_kernel_offsets, augmented_kernel_size

function SplitExplicitAuxiliaryFields(grid::TRG)

    Gᵁ = Field((Face,   Center, Nothing), grid)
    Gⱽ = Field((Center, Face,   Nothing), grid)

    bcs_fc = FieldBoundaryConditions(
            top    = nothing,
            bottom = nothing,
            west   = Gᵁ.boundary_conditions.west,
            east   = Gᵁ.boundary_conditions.east,         
            south  = Gᵁ.boundary_conditions.south,
            north  = ZipperBoundaryCondition()
    )
    
    bcs_cf = FieldBoundaryConditions(
            top    = nothing,
            bottom = nothing,
            west   = Gⱽ.boundary_conditions.west,
            east   = Gⱽ.boundary_conditions.east,         
            south  = Gⱽ.boundary_conditions.south,
            north  = ZipperBoundaryCondition()
    )
    
    Hᶠᶜ = Field((Face,   Center, Nothing), grid; boundary_conditions = bcs_fc)
    Hᶜᶠ = Field((Center, Face,   Nothing), grid; boundary_conditions = bcs_cf)

    calculate_column_height!(Hᶠᶜ, (Face, Center, Center))
    calculate_column_height!(Hᶜᶠ, (Center, Face, Center))

    fill_halo_regions!((Hᶠᶜ, Hᶜᶠ))

    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)

    kernel_parameters = KernelParameters((Nx, Ny + Hy - 1), (0, 0))
    
    return SplitExplicitAuxiliaryFields(Gᵁ, Gⱽ, Hᶠᶜ, Hᶜᶠ, kernel_parameters)
end

@inline tripolar_split_explicit_halos(old_halos, step_halo) = old_halos[1], max(step_halo, old_halos[2]), old_halos[3]

# Internal function for HydrostaticFreeSurfaceModel
function materialize_free_surface(free_surface::SplitExplicitFreeSurface, velocities, grid::TRG)

        settings  = free_surface.settings 

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
