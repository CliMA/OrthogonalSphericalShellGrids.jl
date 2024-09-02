using OrthogonalSphericalShellGrids
using Oceananigans
using Oceananigans: GPU, CPU
using Oceananigans.BoundaryConditions
using Oceananigans.CUDA
using Test

arch = CUDA.has_cuda_gpu() ? GPU() : CPU()

@testset "OrthogonalSphericalShellGrids.jl" begin
    # We probably do not need any unit tests.

    # Test the grid?
    grid = TripolarGrid(arch; size = (10, 10, 1))

    # Test boundary conditions?
    u = XFaceField(grid)
    v = YFaceField(grid)
    c = CenterField(grid)

    set!(u, 1.0)
    set!(v, 1.0)
    set!(c, 1.0)

    fill_halo_regions!(u)
    fill_halo_regions!(v)
    fill_halo_regions!(c)

    @test all(u.data .== 1.0)
end
