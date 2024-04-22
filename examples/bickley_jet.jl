using Oceananigans
using Oceananigans.Units
using Printf
using OrthogonalSphericalShellGrids

Nx = 360
Ny = 180
Nb = 20

underlying_grid = TripolarGrid(size = (Nx, Ny, 1), halo = (5, 5, 5))

bottom_height = zeros(Nx, Ny)

bottom_height[1:Nb+1, end-Nb:end]                .= 1
bottom_height[end-Nb:end, end-Nb:end]            .= 1
bottom_height[(Nx-Nb)÷2:(Nx+Nb)÷2+1, end-Nb:end] .= 1
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_height))

Ψ(y) = - tanh(y) * 10
U(y) = sech(y)^2 

# A sinusoidal tracer
C(y, L) = sin(2π * y / L)

# Slightly off-center vortical perturbations
ψ̃(x, y, ℓ, k) = exp(-(y + ℓ/10)^2 / 2ℓ^2) * cos(k * x) * cos(k * y)

# Vortical velocity fields (ũ, ṽ) = (-∂_y, +∂_x) ψ̃
ũ(x, y, ℓ, k) = + ψ̃(x, y, ℓ, k) * (k * tan(k * y) + y / ℓ^2) 
ṽ(x, y, ℓ, k) = - ψ̃(x, y, ℓ, k) * k * tan(k * x) 

free_surface = SplitExplicitFreeSurface(grid; substeps = 30)

@info "Building a model..."; start=time_ns()

tracer_advection = Oceananigans.Advection.TracerAdvection(WENO(; order = 5), WENO(; order = 5), Centered())
momentum_advection = WENOVectorInvariant(vorticity_order = 5)

model = HydrostaticFreeSurfaceModel(; grid, free_surface,
                                      momentum_advection,
                                      tracer_advection,
                                      buoyancy = nothing,
                                      tracers = :c)

ζ = Oceananigans.Models.HydrostaticFreeSurfaceModels.VerticalVorticityField(model)    

# Parameters
ϵ = 0.1 # perturbation magnitude
ℓ = 0.5 # Gaussian width
k = 2.5 # Sinusoidal wavenumber

dr(x) = deg2rad(x)

# Total initial conditions
uᵢ(x, y, z) = U(dr(y)*8) + ϵ * ũ(dr(x)*2, dr(y)*8, ℓ, k)
vᵢ(x, y, z) = ϵ * ṽ(dr(x)*2, dr(y)*4, ℓ, k)
cᵢ(x, y, z) = C(dr(y)*8, 167.0)

set!(model, u=uᵢ, v=vᵢ, c=cᵢ)

# Δx = minimum_xspacing(grid)
# Δy = minimum_yspacing(grid)

# Δ = 1 / sqrt(1 / Δx^2 + 1 / Δy^2)
# c = sqrt(model.free_surface.gravitational_acceleration)
# Δt = 0.3 * Δ / c

Δt = 1minutes

wizard = TimeStepWizard(cfl=0.3, max_change=1.1, max_Δt=1hour)

simulation = Simulation(model, Δt=Δt, stop_time=15hours)

simulation.output_writers[:surface_tracer] = JLD2OutputWriter(model, merge(model.velocities, model.tracers, (; ζ)),
                                                              filename = "orca025_bickley.jld2", 
                                                              schedule = TimeInterval(1day),
                                                              overwrite_existing = true)

progress(sim) = @info @sprintf("%s with %s, velocity: %.2e %.2e", prettytime(time(sim)), prettytime(sim.Δt), maximum(sim.model.velocities.u), maximum(sim.model.velocities.v)) 

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))
simulation.callbacks[:wizard]   = Callback(wizard,   IterationInterval(10))

# run!(simulation)

# Let's visualize the fields!
