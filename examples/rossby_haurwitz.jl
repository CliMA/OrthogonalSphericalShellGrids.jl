using Oceananigans
using Oceananigans.Units
using Printf
using OrthogonalSphericalShellGrids

Nx = 360
Ny = 180
Nb = 30

underlying_grid = TripolarGrid(size = (Nx, Ny, 1), halo = (5, 5, 5))

bottom_height = zeros(Nx, Ny)

bottom_height[1:Nb+1, end-Nb:end]                .= 1
bottom_height[end-Nb:end, end-Nb:end]            .= 1
bottom_height[(Nx-Nb)÷2:(Nx+Nb)÷2+1, end-Nb:end] .= 1

@show size(underlying_grid)
@show size(bottom_height)

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
                                      coriolis = HydrostaticSphericalCoriolis(),
                                      buoyancy = nothing,
                                      tracers = :c)

K = 7.848e-6
ω = 0
n = 4

R = grid.radius
g = model.free_surface.gravitational_acceleration
Ω = model.coriolis.rotation_rate

A(θ) = ω/2 * (2 * Ω + ω) * cos(θ)^2 + 1/4 * K^2 * cos(θ)^(2*n) * ((n+1) * cos(θ)^2 + (2 * n^2 - n - 2) - 2 * n^2 * sec(θ)^2)
B(θ) = 2 * K * (Ω + ω) * ((n+1) * (n+2))^(-1) * cos(θ)^(n) * ( n^2 + 2*n + 2 - (n+1)^2 * cos(θ)^2) # Why not (n+1)^2 sin(θ)^2 + 1?
C(θ)  = 1/4 * K^2 * cos(θ)^(2 * n) * ( (n+1) * cos(θ)^2 - (n+2))

ψ_function(θ, ϕ) = -R^2 * ω * sin(θ)^2 + R^2 * K * cos(θ)^n * sin(θ) * cos(n*ϕ)

u_function(θ, ϕ) =  R * ω * cos(θ) + R * K * cos(θ)^(n-1) * (n * sin(θ)^2 - cos(θ)^2) * cos(n*ϕ)
v_function(θ, ϕ) = -n * K * R * cos(θ)^(n-1) * sin(θ) * sin(n*ϕ)

h_function(θ, ϕ) = H + R^2/g * (A(θ) + B(θ) * cos(n * ϕ) + C(θ) * cos(2n * ϕ))

rescale¹(λ) = (λ + 180) / 360 * 2π # λ to θ
rescale²(ϕ) = ϕ / 180 * π # θ to ϕ

# Arguments were u(θ, ϕ), λ |-> ϕ, θ |-> ϕ
#=
u₀(λ, ϕ, z) = u_function(rescale²(ϕ), rescale¹(λ))
v₀(λ, ϕ, z) = v_function(rescale²(ϕ), rescale¹(λ))
=#
η₀(λ, ϕ, z) = h_function(rescale²(ϕ), rescale¹(λ))

#=
set!(model, u=u₀, v=v₀, η = η₀)
=#

ψ₀(λ, φ, z) = ψ_function(rescale²(φ), rescale¹(λ))

ψ = Field{Face, Face, Center}(grid)

# Note that set! fills only interior points; to compute u and v we need information in the halo regions.
set!(ψ, ψ₀)

fill_halo_regions!(ψ)

u = Field(- ∂y(ψ))
v = Field(∂x(ψ))

compute!(u)
compute!(v)

fill_halo_regions!((u, v))

set!(model, u = u, v = v, η = η₀)
initialize!(model)
