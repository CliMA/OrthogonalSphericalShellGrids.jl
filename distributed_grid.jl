    using OrthogonalSphericalShellGrids
    using Oceananigans
    using MPI
    MPI.Init()

    include("test/distributed_tests_utils.jl")
    arch = Distributed(CPU(), partition = Partition(2, 2))
    
    distributed_grid = TripolarGrid(arch; size = (100, 100, 1), z = (-1000, 0))
    distributed_grid = mask_singularities(distributed_grid)

    run_tripolar_simulation(distributed_grid)

    if arch.local_rank == 0
        η = reconstruct_global_field(model.free_surface.η)
        u = reconstruct_global_field(model.velocities.u)
        v = reconstruct_global_field(model.velocities.v)

        fill_halo_regions!(η)
        fill_halo_regions!(u)
        fill_halo_regions!(v)
        jldsave("distributed_tripolar.jld2"; η = η.data, u = u.data, v = v.data) 
    end

    MPI.Barrier(MPI.COMM_WORLD)
    MPI.Finalize()
