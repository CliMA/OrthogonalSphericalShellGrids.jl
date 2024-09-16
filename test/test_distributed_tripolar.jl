using MPI

run_distributed_grid = """
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
"""

@testset "Test distributed TripolarGrid..." begin
    write("distributed_grid.jl", run_distributed_grid)

    mpiexec(cmd -> run(`$cmd -n 4 julia --project distributed_grid.jl`))

    arch = CPU()
    
    grid = TripolarGrid(arch; size = (100, 100, 1), z = (-1000, 0))
    λp = grid.conformal_mapping.first_pole_longitude
    φp = grid.conformal_mapping.north_poles_latitude
    
    grid = mask_singularities(grid)

    run_tripolar_simulation(grid)

    # Serial quantities
    us, vs, ws = model.velocities
    ηs = model.free_surface.η

    # Parallel quantities
    up = jldopen("distributed_tripolar.jld2")["u"]
    vp = jldopen("distributed_tripolar.jld2")["u"]
    ηp = jldopen("distributed_tripolar.jld2")["u"]

    @test us.data ≈ up
    @test vs.data ≈ vp
    @test ηs.data ≈ ηp
end