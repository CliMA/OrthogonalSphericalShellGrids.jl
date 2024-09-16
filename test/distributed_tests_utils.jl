using JLD2
using MPI
using Oceananigans.DistributedComputations: reconstruct_global_field

include("dependencies_for_runtests.jl")

# Run the distributed grid simulation and save down reconstructed results
function run_distributed_tripolar_grid(arch, filename)
    distributed_grid = TripolarGrid(arch; size = (100, 100, 1), z = (-1000, 0))
    distributed_grid = mask_singularities(distributed_grid)
    simulation       = run_tripolar_simulation(distributed_grid)

    η = reconstruct_global_field(simulation.model.free_surface.η)
    u = reconstruct_global_field(simulation.model.velocities.u)
    v = reconstruct_global_field(simulation.model.velocities.v)
    c = reconstruct_global_field(simulation.model.velocities.c)

    fill_halo_regions!(η)
    fill_halo_regions!(u)
    fill_halo_regions!(v)
    fill_halo_regions!(c)

    if arch.local_rank == 0
        jldsave(filename; η = interior(η, :, :, 1), u = u.data, v = v.data, c = c.data) 
    end

    MPI.Barrier(MPI.COMM_WORLD)
    MPI.Finalize()

    return nothing
end

# Just a random simulation on a tripolar grid
function run_tripolar_simulation(grid)

    model = HydrostaticFreeSurfaceModel(; grid = grid,
                                          free_surface = SplitExplicitFreeSurface(grid; substeps = 20),
                                          tracers = :c,
                                          buoyancy = nothing, 
                                          tracer_advection = WENO(),
                                          momentum_advection = VectorInvariant(),
                                          coriolis = HydrostaticSphericalCoriolis())

    # Setup the model with a gaussian sea surface height
    # near the physical north poles and one near the equator
    ηᵢ(λ, φ, z) = exp(- (φ - 90)^2 / 10^2) + exp(- φ^2 / 10^2)
    
    set!(model, η = ηᵢ, c = ηᵢ)

    simulation = Simulation(model, Δt = 5minutes, stop_iteration = 100)
    
    run!(simulation)

    return simulation
end