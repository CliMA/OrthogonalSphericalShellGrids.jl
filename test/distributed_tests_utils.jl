
function run_tripolar_simulation(grid)

    model = HydrostaticFreeSurfaceModel(; grid = grid,
                                            free_surface = SplitExplicitFreeSurface(grid; substeps = 20),
                                            tracers = (),
                                            buoyancy = nothing, 
                                            coriolis = HydrostaticSphericalCoriolis())

    # Setup the model with a gaussian profile near the physical north poles
    ηᵢ(λ, φ, z) = exp(- (φ - 90)^2 / 10^2) 

    set!(model, η = (x, y, z) -> exp( - (x - λp)^2 / 10^2 - (y - φp)^2 / 10^2) * 0.1)

    simulation = Simulation(model, Δt = 1minutes, stop_iteration = 100)
    
    outputs = merge(model.velocities, (; η = model.free_surface.η))

    run!(simulation)

    return nothing
end

function mask_singularities(underlying_grid)
    λp = underlying_grid.conformal_mapping.first_pole_longitude
    φp = underlying_grid.conformal_mapping.north_poles_latitude
    
    # We need a bottom height field that ``masks'' the singularities
    bottom_height(λ, φ) = ((abs(λ - λp) < 5)       & (abs(φp - φ) < 5)) |
                          ((abs(λ - λp - 180) < 5) & (abs(φp - φ) < 5)) | (φ < -80) ? 0 : - 1000

    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))

    return grid
end
