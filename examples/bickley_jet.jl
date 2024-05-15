using Oceananigans
using Oceananigans.Units
using Printf
using OrthogonalSphericalShellGrids
using Oceananigans.Utils: get_cartesian_nodes_and_vertices

Nx = 360
Ny = 180
Nb = 40

first_pole_longitude = λ¹ₚ = 45
north_poles_latitude = φₚ  = 25

λ²ₚ = λ¹ₚ + 180

# Build a tripolar grid with singularities at
# (0, -90), (45, 25), (225, 25)
underlying_grid = TripolarGrid(; size = (Nx, Ny, 1), 
                                 halo = (5, 5, 5), 
                                 first_pole_longitude,
                                 north_poles_latitude)

# We need a bottom height field that ``masks'' the singularities
bottom_height(λ, φ) = ((abs(λ - λ¹ₚ) < 5) & (abs(φₚ - φ) < 5)) |
                      ((abs(λ - λ²ₚ) < 5) & (abs(φₚ - φ) < 5)) | (φ < -78) ? 1 : 0

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

Δt = 1minutes

wizard = TimeStepWizard(cfl=0.3, max_change=1.1, max_Δt=3hours)

simulation = Simulation(model, Δt=Δt, stop_time=500days)

simulation.output_writers[:surface_tracer] = JLD2OutputWriter(model, merge(model.velocities, model.tracers, (; ζ)),
                                                              filename = "tripolar_bickley.jld2", 
                                                              schedule = TimeInterval(1day),
                                                              overwrite_existing = true)

progress(sim) = @info @sprintf("%s with %s, velocity: %.2e %.2e", prettytime(time(sim)), prettytime(sim.Δt), maximum(sim.model.velocities.u), maximum(sim.model.velocities.v)) 

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))
simulation.callbacks[:wizard]   = Callback(wizard,   IterationInterval(10))

run!(simulation)

# Let's visualize the fields!
ζ = FieldTimeSeries("orca025_bickley.jld2", "ζ")
c = FieldTimeSeries("orca025_bickley.jld2", "c")

Nt = length(ζ.times)

iter = Observable(1)

# Lift the correct time step
ζi = @lift(interior(ζ[$iter], :, :, 1))
ci = @lift(interior(c[$iter], :, :, 1))

# retrieve the Face-Face nodes in a Cartesian coordinate system
cartesian_nodes, _ = get_cartesian_nodes_and_vertices(underlying_grid, Face(), Face(), Center())
xF, yF, zF = cartesian_nodes

# retrieve the Center-Center nodes in a Cartesian coordinate system
cartesian_nodes, _ = get_cartesian_nodes_and_vertices(underlying_grid, Center(), Center(), Center())
xC, yC, zC = cartesian_nodes

fig = Figure()

ax = LScene(fig[1, 1]; show_axis = false)
surface!(ax, xF, yF, zF, color = ζi, colorrange = (-5e-6, 5e-6), colormap = :bwr)

ax = LScene(fig[1, 2]; show_axis = false)
surface!(ax, xC, yC, zC, color = ci, colorrange = (-1, 1), colormap = :magma)

GLMakie.record(fig, "spherical_bickley_with_continents.mp4", 1:Nt, framerate = 5) do i
    @info "printing iteration $i of $Nt"
    iter[] = i
end

# For fun, since the data on a Tripolar grid is topologically a matrix, let's look at 
# how it looks projected on a plane! 
# To make sure everything is working as expected let's plot the whole data including halo regions

fig = Figure(size = (1500, 800))

# Lift the correct time step
ζi = @lift begin
    z = ζ[$iter].data[:, :, 1]
    z[z .== 0] .= NaN
    z
end

ci = @lift begin 
    t = c[$iter].data[:, :, 1]
    t[t .== 0] .= NaN
    t
end

ax = Axis(fig[1, 1])
heatmap!(ax, ζi, colorrange = (-5e-6, 5e-6), colormap = :bwr)

ax = Axis(fig[1, 2])
heatmap!(ax, ci, colorrange = (-1, 1), colormap = :magma)

GLMakie.record(fig, "spherical_bickley_on_a_plane.mp4", 10:Nt, framerate = 5) do i
    @info "printing iteration $i of $Nt"
    iter[] = i
end