include("dependencies_for_runtests.jl")

@testset "Unit tests..." begin
    grid = TripolarGrid(size = (4, 5, 1), z = (0, 1), 
                        first_pole_longitude = 75, 
                        north_poles_latitude = 35,
                        southermost_latitude = -80)

    @test grid isa TripolarGrid

    @test grid.Nx == 4
    @test grid.Ny == 5
    @test grid.Nz == 1

    @test grid.conformal_mapping.first_pole_longitude == 75
    @test grid.conformal_mapping.north_poles_latitude == 35
    @test grid.conformal_mapping.southermost_latitude == -80

    λᶜᶜᵃ = λnodes(grid, Center(), Center())
    φᶜᶜᵃ = φnodes(grid, Center(), Center())

    min_Δφ = minimum(φᶜᶜᵃ[:, 2] .- φᶜᶜᵃ[:, 1])

    @test minimum(λᶜᶜᵃ) ≥ 0
    @test maximum(λᶜᶜᵃ) ≤ 360
    @test maximum(φᶜᶜᵃ) ≤ 90

    # The minimum latitude is not exactly the southermost latitude because the grid 
    # undulates slightly to maintain the same analytical description in the whole sphere
    # (i.e. constant latitude lines do not exist anywhere in this grid)
    @test minimum(φᶜᶜᶜ .+ min_Δφ / 10) ≥ grid.conformal_mapping.southermost_latitude 
end

include("test_tripolar_grid.jl")
include("test_zipper_boundary_conditions.jl")

