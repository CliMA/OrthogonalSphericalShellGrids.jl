using OrthogonalSphericalShellGrids
using Oceananigans
using Oceananigans: GPU, CPU
using Oceananigans.BoundaryConditions
using OrthogonalSphericalShellGrids: get_cartesian_nodes_and_vertices
using Oceananigans.CUDA
using Test

arch = CUDA.has_cuda_gpu() ? GPU() : CPU()
