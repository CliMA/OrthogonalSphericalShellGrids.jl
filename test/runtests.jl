include("dependencies_for_runtests.jl")

@testset "OrthogonalSphericalShellGrids.jl" begin
    # Test the grid?
    grid = TripolarGrid(arch; size = (10, 10, 1))

    # Test boundary conditions?
    c = CenterField(grid)

    set!(c, 1.0)
    fill_halo_regions!(c)

    @test all(c.data .== 1.0) 
end
