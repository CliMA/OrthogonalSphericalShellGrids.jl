using OrthogonalSphericalShellGrids
using Oceananigans
using Oceananigans.Grids: halo_size, φnodes, λnodes
using Oceananigans.Utils
using Oceananigans.BoundaryConditions
using OrthogonalSphericalShellGrids: get_cartesian_nodes_and_vertices
using Oceananigans.CUDA
using Test

using KernelAbstractions: @kernel, @index

arch = CUDA.has_cuda_gpu() ? GPU() : CPU()
