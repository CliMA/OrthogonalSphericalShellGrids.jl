include("dependencies_for_runtests.jl")
include("distributed_tests_utils.jl")
using MPI

tripolar_boundary_conditions = """
    using MPI
    MPI.Init()

    include("distributed_tests_utils.jl")

    arch = Distributed(CPU(), partition = Partition(2, 2))
    grid = TripolarGrid(arch; size = (20, 20, 1), z = (-1000, 0))

    u = XFaceField(grid)
    v = YFaceField(grid)
    c = CenterField(grid)

    set!(u, (x, y, z) -> y)
    set!(v, (x, y, z) -> y)
    set!(c, (x, y, z) -> y)

    fill_halo_regions!((u, v, c))

    jldopen("distributed_tripolar_boundary_conditions_" * string(arch.local_rank) * ".jld2", "w") do file
        file["u"] = u.data
        file["v"] = v.data
        file["c"] = c.data
    end
"""

@testset "Test distributed TripolarGrid boundary conditions..." begin
    # Run the serial computation    
    grid = TripolarGrid(size = (20, 20, 1), z = (-1000, 0))

    u = XFaceField(grid)
    v = YFaceField(grid)
    c = CenterField(grid)

    set!(u, (x, y, z) -> y)
    set!(v, (x, y, z) -> y)
    set!(c, (x, y, z) -> y)

    fill_halo_regions!((u, v, c))
    
    write("distributed_tests.jl", tripolar_boundary_conditions)
    mpiexec(cmd -> run(`$cmd -n 4 julia --project distributed_tests.jl`))
    rm("distributed_tests.jl")

    # Retrieve Parallel quantities from rank 1 (the north-west rank)
    up1 = jldopen("distributed_tripolar_boundary_conditions_1.jld2")["u"];
    vp1 = jldopen("distributed_tripolar_boundary_conditions_1.jld2")["v"];
    cp1 = jldopen("distributed_tripolar_boundary_conditions_1.jld2")["c"];

    # Retrieve Parallel quantities from rank 3 (the north-east rank)
    up3 = jldopen("distributed_tripolar_boundary_conditions_3.jld2")["u"];
    vp3 = jldopen("distributed_tripolar_boundary_conditions_3.jld2")["v"];
    cp3 = jldopen("distributed_tripolar_boundary_conditions_3.jld2")["c"];

    # @test u.data[-2:14, 7:end-1, 1] ≈ up1.parent[2:end, 1:end-1, 5]
    @test v.data[-3:14, 7:end-1, 1] ≈ vp1.parent[:,     1:end-1, 5]
    @test c.data[-3:14, 7:end-1, 1] ≈ cp1.parent[:,     1:end-1, 5]

    # @test u.data[8:end, 7:end-1, 1] ≈ up3.parent[2:end, 1:end-1, 5]
    @test v.data[7:end, 7:end-1, 1] ≈ vp3.parent[:,     1:end-1, 5]
    @test c.data[7:end, 7:end-1, 1] ≈ cp3.parent[:,     1:end-1, 5]
end

run_slab_distributed_grid = """
    using MPI
    MPI.Init()

    include("distributed_tests_utils.jl")
    arch = Distributed(CPU(), partition = Partition(1, 4))
    run_distributed_tripolar_grid(arch, "distributed_yslab_tripolar.jld2")
"""

run_pencil_distributed_grid = """
    using MPI
    MPI.Init()

    include("distributed_tests_utils.jl")
    arch = Distributed(CPU(), partition = Partition(2, 2))
    run_distributed_tripolar_grid(arch, "distributed_pencil_tripolar.jld2")
"""

@testset "Test distributed TripolarGrid simulations..." begin
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

    @test interior(us, :, :, 1) ≈ up_slab
    @test interior(vs, :, :, 1) ≈ vp_slab
    @test interior(cs, :, :, 1) ≈ cp_slab
    @test interior(ηs, :, :, 1) ≈ ηp_slab

    @test interior(us, :, :, 1) ≈ up_pencil
    @test interior(vs, :, :, 1) ≈ vp_pencil
    @test interior(cs, :, :, 1) ≈ cp_pencil
    @test interior(ηs, :, :, 1) ≈ ηp_pencil
end