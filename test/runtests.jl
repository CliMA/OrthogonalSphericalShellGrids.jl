include("dependencies_for_runtests.jl")

@testset "Unit tests..." begin
    grid = TripolarGrid(size = (360, 180, 1), z = (0, 1), 
                        first_pole_longitude = 75, 
                        north_poles_latitude = 35,
                        southermost_latitude = -80)

    @test grid isa TripolarGrid

    @test grid.Nx == 360
    @test grid.Ny == 180
    @test grid.Nz == 1

    @test grid.conformal_mapping.first_pole_longitude == 75
    @test grid.conformal_mapping.north_poles_latitude == 35
    @test grid.conformal_mapping.southermost_latitude == -80

    min_Δφ = minimum(grid.φᶜᶜᵃ[:, 2] .- grid.φᶜᶜᵃ[:, 1])

    @test minimum(grid.λᶜᶜᵃ) ≥ 0
    @test maximum(grid.λᶜᶜᵃ) ≤ 360
    @test maximum(grid.φᶜᶜᵃ) ≤ 90

    # The minimum latitude is not exactly the southermost latitude because the grid 
    # ondulates slightly to maintain the same analytical description in the whole sphere
    # (i.e. constant latitude lines do not exist anywhere in this grid)
    @test minimum(grid.φᶜᶜᵃ .+ min_Δφ / 10) ≥ grid.conformal_mapping.southermost_latitude 
end

include("test_tripolar_grid.jl")
include("test_zipper_boundary_conditions.jl")

