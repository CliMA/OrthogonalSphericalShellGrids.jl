include("dependencies_for_runtests.jl")
include("distributed_tests_utils.jl")
using MPI

@testset "Test distributed TripolarGrid boundary conditions..." begin
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

    write("distributed_tests.jl", tripolar_boundary_conditions)
    mpiexec(cmd -> run(`$cmd -n 4 julia --project distributed_tests.jl`))
    rm("distributed_tests.jl")

    # Run the serial computation    
    grid = TripolarGrid(size = (20, 20, 1), z = (-1000, 0))

    u = XFaceField(grid)
    v = YFaceField(grid)
    c = CenterField(grid)

    set!(u, (x, y, z) -> y)
    set!(v, (x, y, z) -> y)
    set!(c, (x, y, z) -> y)

    fill_halo_regions!((u, v, c))
    
    # Retrieve Parallel quantities from rank 1 (the north-west rank)
    up1 = jldopen("distributed_tripolar_boundary_conditions_1.jld2")["u"];
    vp1 = jldopen("distributed_tripolar_boundary_conditions_1.jld2")["v"];
    cp1 = jldopen("distributed_tripolar_boundary_conditions_1.jld2")["c"];

    # Retrieve Parallel quantities from rank 3 (the north-east rank)
    up3 = jldopen("distributed_tripolar_boundary_conditions_3.jld2")["u"];
    vp3 = jldopen("distributed_tripolar_boundary_conditions_3.jld2")["v"];
    cp3 = jldopen("distributed_tripolar_boundary_conditions_3.jld2")["c"];

    @test u.data[-3:14, 7:end, 1] ≈ up1[:, :, 1].parent
    @test v.data[-3:14, 7:end, 1] ≈ vp1[:, :, 1].parent
    @test c.data[-3:14, 7:end, 1] ≈ cp1[:, :, 1].parent

    @test us.data[7:end, 7:end, 1] ≈ up3[:, :, 1].parent
    @test vs.data[7:end, 7:end, 1] ≈ vp3[:, :, 1].parent
    @test cs.data[7:end, 7:end, 1] ≈ cp3[:, :, 1].parent
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
    up_xslab = jldopen("distributed_slab_tripolar.jld2")["u"]
    vp_xslab = jldopen("distributed_slab_tripolar.jld2")["v"]
    ηp_xslab = jldopen("distributed_slab_tripolar.jld2")["η"]
    cp_xslab = jldopen("distributed_slab_tripolar.jld2")["c"]

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