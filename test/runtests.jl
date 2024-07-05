using OrthogonalSphericalShellGrids
using OrthogonalSphericalShellGrids.Oceananigans
using Oceananigans.CUDA
using Test

arch = CUDA.has_cuda_gpu() ? GPU() : CPU()

@testset "OrthogonalSphericalShellGrids.jl" begin
    # We probably do not need any unit tests.

    # Test the grid?
    grid = TripolarGrid(arch; size = (10, 10, 1))
    # Test boundary conditions?    
    
end
