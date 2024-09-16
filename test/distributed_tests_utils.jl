using OrthogonalSphericalShellGrids
using Oceananigans
using Oceananigans.Units
using Oceananigans.BoundaryConditions
using Oceananigans.DistributedComputations: reconstruct_global_field
using JLD2
using MPI

function run_distributed_tripolar_grid(arch, filename)
    distributed_grid = TripolarGrid(arch; size = (100, 100, 1), z = (-1000, 0))
    distributed_grid = mask_singularities(distributed_grid)
    simulation       = run_tripolar_simulation(distributed_grid)

    η = reconstruct_global_field(simulation.model.free_surface.η)
    u = reconstruct_global_field(simulation.model.velocities.u)
    v = reconstruct_global_field(simulation.model.velocities.v)

    fill_halo_regions!(η)
    fill_halo_regions!(u)
    fill_halo_regions!(v)

    if arch.local_rank == 0
        jldsave(filename; η = interior(η, :, :, 1), u = u.data, v = v.data) 
    end

    MPI.Barrier(MPI.COMM_WORLD)
    MPI.Finalize()

    return nothing
end

function run_tripolar_simulation(grid)

    model = HydrostaticFreeSurfaceModel(; grid = grid,
                                          free_surface = SplitExplicitFreeSurface(grid; substeps = 20),
                                          tracers = (),
                                          buoyancy = nothing, 
                                          momentum_advection = VectorInvariant(),
                                          coriolis = HydrostaticSphericalCoriolis())

    # Setup the model with a gaussian sea surface height
    # near the physical north poles and one near the equator
    ηᵢ(λ, φ, z) = exp(- (φ - 90)^2 / 10^2) + exp(- φ^2 / 10^2)

    set!(model, η = ηᵢ)

    simulation = Simulation(model, Δt = 5minutes, stop_iteration = 100)
    
    run!(simulation)

    return simulation
end

function mask_singularities(underlying_grid)
    λp = underlying_grid.conformal_mapping.first_pole_longitude
    φp = underlying_grid.conformal_mapping.north_poles_latitude
    
    # We need a bottom height field that ``masks'' the singularities
    bottom_height(λ, φ) = ((abs(λ - λp) < 5)       & (abs(φp - φ) < 5)) |
                          ((abs(λ - λp - 180) < 5) & (abs(φp - φ) < 5)) | (φ < -80) ? 0 : - 1000

    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))

    return grid
end
