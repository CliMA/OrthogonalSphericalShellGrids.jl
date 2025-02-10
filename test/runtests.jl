include("dependencies_for_runtests.jl")
include("distributed_tests_utils.jl")

import Oceananigans.Utils: contiguousrange
using Oceananigans.Utils: KernelParameters

contiguousrange(::KernelParameters{spec, offset}) where {spec, offset} = contiguousrange(spec, offset)

@testset "Unit tests..." begin
    for FT in (Float32, Float64)
        grid = TripolarGrid(CPU(), FT; 
                            size = (4, 5, 1), z = (0, 1), 
                            first_pole_longitude = 75, 
                            north_poles_latitude = 35,
                            southernmost_latitude = -80)

        @test eltype(grid) == FT
        @test grid isa TripolarGrid

        @test grid.Nx == 4
        @test grid.Ny == 5
        @test grid.Nz == 1

        @test grid.conformal_mapping.first_pole_longitude == 75
        @test grid.conformal_mapping.north_poles_latitude == 35
        @test grid.conformal_mapping.southernmost_latitude == -80

        λᶜᶜᵃ = λnodes(grid, Center(), Center())
        φᶜᶜᵃ = φnodes(grid, Center(), Center())

        min_Δφ = minimum(φᶜᶜᵃ[:, 2] .- φᶜᶜᵃ[:, 1])

        @test minimum(λᶜᶜᵃ) ≥ 0
        @test maximum(λᶜᶜᵃ) ≤ 360
        @test maximum(φᶜᶜᵃ) ≤ 90

        # The minimum latitude is not exactly the southermost latitude because the grid 
        # undulates slightly to maintain the same analytical description in the whole sphere
        # (i.e. constant latitude lines do not exist anywhere in this grid)
        @test minimum(φᶜᶜᵃ .+ min_Δφ / 10) ≥ grid.conformal_mapping.southernmost_latitude 
    end
end

include("test_tripolar_grid.jl")
include("test_zipper_boundary_conditions.jl")

@testset "Model tests..." begin
    grid = TripolarGrid(size = (10, 10, 1))

    # Wrong free surface
    @test_throws ArgumentError HydrostaticFreeSurfaceModel(; grid)

    free_surface = SplitExplicitFreeSurface(grid; substeps = 12)
    model = HydrostaticFreeSurfaceModel(; grid, free_surface)

    # Tests the grid has been extended
    η = model.free_surface.η
    P = model.free_surface.kernel_parameters

    range = contiguousrange(P)

    # Should have extended halos in the north
    Hx, Hy, _ = halo_size(η.grid)
    Nx, Ny, _ = size(grid)

    @test P isa KernelParameters
    @test range[1] == 1:Nx
    @test range[2] == 1:Ny+Hy-1 
    
    @test Hx == halo_size(grid, 1)
    @test Hy != halo_size(grid, 2)
    @test Hy == length(free_surface.substepping.averaging_weights) + 1
    
    @test begin
        time_step!(model, 1.0)
        true
    end
end