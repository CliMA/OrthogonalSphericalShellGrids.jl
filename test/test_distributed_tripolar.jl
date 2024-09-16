include("dependencies_for_runtests.jl")
include("distributed_tests_utils.jl")
using MPI

run_slab_distributed_grid = """
    using Oceananigans
    using MPI
    MPI.Init()

    include("distributed_tests_utils.jl")
    arch = Distributed(CPU(), partition = Partition(1, 4))
    run_distributed_tripolar_grid(arch, "distributed_slab_tripolar.jld2")
"""

run_pencil_distributed_grid = """
    using Oceananigans
    using MPI
    MPI.Init()

    include("distributed_tests_utils.jl")
    arch = Distributed(CPU(), partition = Partition(2, 2))
    run_distributed_tripolar_grid(arch, "distributed_pencil_tripolar.jld2")
"""

@testset "Test distributed TripolarGrid..." begin
    # Run the distributed grid simulation
    write("distributed_tests.jl", run_slab_distributed_grid)
    mpiexec(cmd -> run(`$cmd -n 4 julia --project distributed_tests.jl`))
    rm("distributed_tests.jl")

    write("distributed_tests.jl", run_pencil_distributed_grid)
    mpiexec(cmd -> run(`$cmd -n 4 julia --project distributed_tests.jl`))
    rm("distributed_tests.jl")

    # Run the serial computation    
    grid = TripolarGrid(size = (100, 100, 1), z = (-1000, 0))
    grid = mask_singularities(grid)

    simulation = run_tripolar_simulation(grid)

    # Retrieve Serial quantities
    us, vs, ws = simulation.model.velocities
    cs = simulation.model.tracers.c
    ηs = simulation.model.free_surface.η

    # Retrieve Parallel quantities
    up_slab = jldopen("distributed_slab_tripolar.jld2")["u"]
    vp_slab = jldopen("distributed_slab_tripolar.jld2")["v"]
    ηp_slab = jldopen("distributed_slab_tripolar.jld2")["η"]
    cp_slab = jldopen("distributed_slab_tripolar.jld2")["c"]

    up_pencil = jldopen("distributed_pencil_tripolar.jld2")["u"]
    vp_pencil = jldopen("distributed_pencil_tripolar.jld2")["v"]
    ηp_pencil = jldopen("distributed_pencil_tripolar.jld2")["η"]
    cp_pencil = jldopen("distributed_pencil_tripolar.jld2")["c"]

    @test us.data ≈ up_slab
    @test vs.data ≈ vp_slab
    @test cs.data ≈ cp_slab
    @test interior(ηs, :, :, 1) ≈ ηp_slab

    @test us.data ≈ up_pencil
    @test vs.data ≈ vp_pencil
    @test cs.data ≈ cp_pencil
    @test interior(ηs, :, :, 1) ≈ ηp_pencil
end