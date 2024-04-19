"""
    struct WarpedLatitudeLongitude

A struct representing a warped latitude-longitude grid.

TODO: put here information about the grid, i.e.: 

1) north pole latitude and longitude
2) functions used to construct the Grid
3) Numerical discretization used to construct the Grid
4) Last great circle size in degrees
"""
struct Tripolar end

"""
    compute_coords!(jnum, xnum, ynum, Δλᶠᵃᵃ, Jeq, f_interpolator, g_interpolator)

Compute the coordinates for an orthogonal spherical shell grid.

# Arguments
- `jnum`: An array to store the computed values of `jnum`.
- `xnum`: An array to store the computed values of `xnum`.
- `ynum`: An array to store the computed values of `ynum`.
- `Δλᶠᵃᵃ`: The angular step size.
- `Jeq`: The value of j at the equator.
- `f_interpolator`: A function that interpolates the value of f.
- `g_interpolator`: A function that interpolates the value of g.

# Details
This function computes the coordinates for an orthogonal spherical shell grid using the given parameters. 
It uses a secant root finding method to find the value of `jnum` and an Adams-Bashforth-2 integrator to find the perpendicular to the circle.
"""
@kernel function compute_tripolar_coords!(jnum, xnum, ynum, λᵢ, Δλ, Δj, Jeq, Nφ, a_interpolator, b_interpolator, c_interpolator)
    i = @index(Global, Linear)
    N = size(xnum, 2)
    @inbounds begin
        h = (λᵢ - Δλ * i) * 2π / 360
        xnum[i, 1], ynum[i, 1] = cos(h), sin(h) # Starting always from a circumpherence at the equator
        Δx = xnum[i, 1] / N
        xnum[i, 2] = xnum[i, 1] - Δx
        ynum[i, 2] = ynum[i, 1] - Δx * tan(h)
        for n in 3:N
            # Great circles
            func(x) = (xnum[i, n-1] / a_interpolator(x)) ^2 + (ynum[i, n-1] / b_interpolator(x))^2 - 1 
            jnum[i, n-1] = bisection_root_find(func, Jeq-1.0, Nφ+1-Δj, Δj)
            xnum[i, n] = xnum[i, n-1] - Δx
            # Adams-Bashforth-2 integrator to find the perpendicular to the circle
            Gnew = ynum[i, n-1] * a_interpolator(jnum[i, n-1])^2 / b_interpolator(jnum[i, n-1])^2 / xnum[i, n-1]
            Gold = ynum[i, n-2] * a_interpolator(jnum[i, n-2])^2 / b_interpolator(jnum[i, n-2])^2 / xnum[i, n-2]

            ynum[i, n] = ynum[i, n-1] - Δx * (1.5 * Gnew - 0.5 * Gold)
        end
        @show i
    end
end

@inline tripolar_stretching_function(φ; d = 0.4) = (φ / 90)^2 * d

@inline cosine_a_curve(φ) = - equator_fcurve(φ) 
@inline cosine_b_curve(φ; d = 145) = - equator_fcurve(φ) + ifelse(φ > 0, tripolar_stretching_function(φ; d), 0)

@inline zero_c_curve(φ) = 0

const TripolarGrid{FT, TX, TY, TZ, A, R, FR, Arch} = OrthogonalSphericalShellGrid{FT, TX, TY, TZ, A, R, FR, <:Tripolar, Arch}

"""
    TripolarGrid(arch = CPU(), FT::DataType = Float64; 
                 size, 
                 southermost_latitude = -75, 
                 halo        = (4, 4, 4), 
                 radius      = R_Earth, 
                 z           = (0, 1),
                 singularity_longitude = 230,
                 f_curve     = quadratic_f_curve,
                 g_curve     = quadratic_g_curve)

Constructs a tripolar grid on a spherical shell.

Positional Arguments
====================

- `arch`: The architecture to use for the grid. Default is `CPU()`.
- `FT::DataType`: The data type to use for the grid. Default is `Float64`.

Keyword Arguments
=================

- `size`: The number of cells in the (longitude, latitude, z) dimensions.
- `southermost_latitude`: The southernmost latitude of the grid. Default is -75.
- `halo`: The halo size in the (longitude, latitude, z) dimensions. Default is (4, 4, 4).
- `radius`: The radius of the spherical shell. Default is `R_Earth`.
- `z`: The z-coordinate range of the grid. Default is (0, 1).
- `singularity_longitude`: The longitude at which the grid has a singularity. Default is 230.
- `f_curve`: The function to compute the f-curve for the grid. Default is `quadratic_f_curve`.
- `g_curve`: The function to compute the g-curve for the grid. Default is `quadratic_g_curve`.

Returns
========

A `OrthogonalSphericalShellGrid` object representing a tripolar grid on the sphere
"""
function TripolarGrid(arch = CPU(), FT::DataType = Float64; 
                      size, 
                      southermost_latitude = -85, 
                      halo                 = (4, 4, 4), 
                      radius               = R_Earth, 
                      z                    = (0, 1),
                      poles_latitude       = 45,
                      first_pole_longitude = 75,    # The second pole will be at `λ = first_pole_longitude + 180ᵒ`
                      Nproc                = 10000, 
                      Nnum                 = 10000, 
                      a_curve              = cosine_a_curve,
                      initial_b_curve      = cosine_b_curve,
                      c_curve              = zero_c_curve)

    # For now, only for domains Periodic in λ (from -180 to 180 degrees) and Bounded in φ.
    # φ has to reach the north pole.`
    # For all the rest we can easily use a `LatitudeLongitudeGrid` without warping
    final_b    = sqrt((tan((90 - poles_latitude) / 360 * π))^2)
    b_curve(φ) = initial_b_curve(φ; d = final_b)

    latitude  = (southermost_latitude, 90)
    longitude = (-180, 180) 
    
    Nλ, Nφ, Nz = size
    Hλ, Hφ, Hz = halo

    Lφ, φᵃᶠᵃ, φᵃᶜᵃ, Δφᵃᶠᵃ, Δφᵃᶜᵃ = generate_coordinate(FT, Bounded(),  Nφ, Hφ, latitude,  :φ, CPU())
    Lλ, λᶠᵃᵃ, λᶜᵃᵃ, Δλᶠᵃᵃ, Δλᶜᵃᵃ = generate_coordinate(FT, Periodic(), Nλ, Hλ, longitude, :λ, CPU())
    Lz, zᵃᵃᶠ, zᵃᵃᶜ, Δzᵃᵃᶠ, Δzᵃᵃᶜ = generate_coordinate(FT, Bounded(),  Nz, Hz, z,         :z, CPU())

    λFF = zeros(Nλ, Nφ)
    φFF = zeros(Nλ, Nφ)
    λFC = zeros(Nλ, Nφ)
    φFC = zeros(Nλ, Nφ)

    λCF = zeros(Nλ, Nφ)
    φCF = zeros(Nλ, Nφ)
    λCC = zeros(Nλ, Nφ)
    φCC = zeros(Nλ, Nφ)

    # Identify equator line 
    J = Ref(0)
    for j in 1:Nφ+1
        if φᵃᶠᵃ[j] < 0
            J[] = j
        end
    end

    Jeq = J[] + 1

    aⱼ = zeros(1:Nproc+1)
    bⱼ = zeros(1:Nproc+1)
    cⱼ = zeros(1:Nproc+1)

    φproc = range(southermost_latitude, 90, length = Nproc) 
    
    # calculate the eccentricities of the ellipse
    for (j, φ) in enumerate(φproc)
        aⱼ[j] = a_curve(φ)
        bⱼ[j] = b_curve(φ) 
        cⱼ[j] = c_curve(φ) 
    end

    fx = Float64.(collect(1:Nproc) ./ Nproc * (Nφ + 1))

    a_interpolator(j) = linear_interpolate(j, fx, aⱼ)
    b_interpolator(j) = linear_interpolate(j, fx, bⱼ)
    c_interpolator(j) = linear_interpolate(j, fx, cⱼ)

    xnum = zeros(1:Nλ+1, Nnum)
    ynum = zeros(1:Nλ+1, Nnum)
    jnum = zeros(1:Nλ+1, Nnum)

    # X - Face coordinates
    λ₀ = 90 # ᵒ degrees  

    loop! = compute_tripolar_coords!(device(CPU()), min(256, Nλ+1), Nλ+1)
    loop!(jnum, xnum, ynum, λ₀, Δλᶠᵃᵃ, 1/Nnum, Jeq, Nφ, a_interpolator, b_interpolator, c_interpolator) 

    # Face - Face 
    loop! = _compute_coordinates!(device(CPU()), (16, 16), (Nλ, Nφ))
    loop!(λFF, φFF, Jeq, λ₀, Δλᶠᵃᵃ, φᵃᶠᵃ, a_curve, xnum, ynum, jnum, Nλ)
    
    # Face - Center 
    loop! = _compute_coordinates!(device(CPU()), (16, 16), (Nλ, Nφ))
    loop!(λFC, φFC, Jeq, λ₀, Δλᶠᵃᵃ, φᵃᶜᵃ, a_curve, xnum, ynum, jnum, Nλ)
    
    # X - Center coordinates
    λ₀ = 90 + Δλᶜᵃᵃ / 2 # ᵒ degrees  

    loop! = compute_tripolar_coords!(device(CPU()), min(256, Nλ+1), Nλ+1)
    loop!(jnum, xnum, ynum, λ₀, Δλᶜᵃᵃ, 1/Nnum, Jeq, Nφ, a_interpolator, b_interpolator, c_interpolator) 
    
    # Face - Face 
    loop! = _compute_coordinates!(device(CPU()), (16, 16), (Nλ, Nφ))
    loop!(λCF, φCF, Jeq, λ₀, Δλᶜᵃᵃ, φᵃᶠᵃ, a_curve, xnum, ynum, jnum, Nλ)
    
    # Face - Center 
    loop! = _compute_coordinates!(device(CPU()), (16, 16), (Nλ, Nφ))
    loop!(λCC, φCC, Jeq, λ₀, Δλᶜᵃᵃ, φᵃᶜᵃ, a_curve, xnum, ynum, jnum, Nλ)
    
    Nx = Nλ
    Ny = Nφ

    # Metrics
    for λ in (λFF, λFC, λCF, λCC)
        λ .+= first_pole_longitude 
        λ .=  convert_to_0_360.(λ)
    end

    # Helper grid to fill halo 
    grid = RectilinearGrid(; size = (Nx, Ny, 1), halo, topology = (Periodic, RightConnected, Bounded), z = (0, 1), x = (0, 1), y = (0, 1))

    default_boundary_conditions = FieldBoundaryConditions(north  = ZipperBoundaryCondition(),
                                                          south  = FluxBoundaryCondition(nothing),
                                                          west   = Oceananigans.PeriodicBoundaryCondition(),
                                                          east   = Oceananigans.PeriodicBoundaryCondition(),
                                                          top    = nothing,
                                                          bottom = nothing)
                                                          
    lFF = Field((Face, Face, Center), grid; boundary_conditions = default_boundary_conditions)
    pFF = Field((Face, Face, Center), grid; boundary_conditions = default_boundary_conditions)

    lFC = Field((Face, Center, Center), grid; boundary_conditions = default_boundary_conditions)
    pFC = Field((Face, Center, Center), grid; boundary_conditions = default_boundary_conditions)
    
    lCF = Field((Center, Face, Center), grid; boundary_conditions = default_boundary_conditions)
    pCF = Field((Center, Face, Center), grid; boundary_conditions = default_boundary_conditions)

    lCC = Field((Center, Center, Center), grid; boundary_conditions = default_boundary_conditions)
    pCC = Field((Center, Center, Center), grid; boundary_conditions = default_boundary_conditions)

    set!(lFF, λFF)
    set!(pFF, φFF)

    set!(lFC, λFC)
    set!(pFC, φFC)

    set!(lCF, λCF)
    set!(pCF, φCF)

    set!(lCC, λCC)
    set!(pCC, φCC)

    fill_halo_regions!((lFF, pFF, lFC, pFC, lCF, pCF, lCC, pCC))

    λᶠᶠᵃ = lFF.data[:, :, 1]
    φᶠᶠᵃ = pFF.data[:, :, 1]

    λᶠᶜᵃ = lFC.data[:, :, 1]
    φᶠᶜᵃ = pFC.data[:, :, 1]

    λᶜᶠᵃ = lCF.data[:, :, 1]
    φᶜᶠᵃ = pCF.data[:, :, 1]

    λᶜᶜᵃ = lCC.data[:, :, 1]
    φᶜᶜᵃ = pCC.data[:, :, 1]

    # Metrics
    Δxᶜᶜᵃ = zeros(Nx, Ny)
    Δxᶠᶜᵃ = zeros(Nx, Ny)
    Δxᶜᶠᵃ = zeros(Nx, Ny)
    Δxᶠᶠᵃ = zeros(Nx, Ny)

    Δyᶜᶜᵃ = zeros(Nx, Ny)
    Δyᶠᶜᵃ = zeros(Nx, Ny)
    Δyᶜᶠᵃ = zeros(Nx, Ny)
    Δyᶠᶠᵃ = zeros(Nx, Ny)

    Azᶜᶜᵃ = zeros(Nx, Ny)
    Azᶠᶜᵃ = zeros(Nx, Ny)
    Azᶜᶠᵃ = zeros(Nx, Ny)
    Azᶠᶠᵃ = zeros(Nx, Ny)

    loop! = _calculate_metrics!(device(CPU()), (16, 16), (Nx, Ny))

    loop!(Δxᶠᶜᵃ, Δxᶜᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
          Δyᶠᶜᵃ, Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶠᵃ,
          Azᶠᶜᵃ, Azᶜᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ,
          λᶠᶜᵃ, λᶜᶜᵃ, λᶜᶠᵃ, λᶠᶠᵃ,
          φᶠᶜᵃ, φᶜᶜᵃ, φᶜᶠᵃ, φᶠᶠᵃ,
          radius)

    # Metrics fields to fill halos
    FF = Field((Face, Face, Center),     grid; boundary_conditions = default_boundary_conditions)
    FC = Field((Face, Center, Center),   grid; boundary_conditions = default_boundary_conditions)
    CF = Field((Center, Face, Center),   grid; boundary_conditions = default_boundary_conditions)
    CC = Field((Center, Center, Center), grid; boundary_conditions = default_boundary_conditions)

    # Fill all periodic halos
    set!(FF, Δxᶠᶠᵃ); set!(CF, Δxᶜᶠᵃ); set!(FC, Δxᶠᶜᵃ); set!(CC, Δxᶜᶜᵃ); 
    fill_halo_regions!((FF, CF, FC, CC))
    Δxᶠᶠᵃ = FF.data[:, :, 1]; 
    Δxᶜᶠᵃ = CF.data[:, :, 1]; 
    Δxᶠᶜᵃ = FC.data[:, :, 1]; 
    Δxᶜᶜᵃ = CC.data[:, :, 1]; 
    set!(FF, Δyᶠᶠᵃ); set!(CF, Δyᶜᶠᵃ); set!(FC, Δyᶠᶜᵃ); set!(CC, Δyᶜᶜᵃ); 
    fill_halo_regions!((FF, CF, FC, CC))
    Δyᶠᶠᵃ = FF.data[:, :, 1]; 
    Δyᶜᶠᵃ = CF.data[:, :, 1]; 
    Δyᶠᶜᵃ = FC.data[:, :, 1]; 
    Δyᶜᶜᵃ = CC.data[:, :, 1]; 
    set!(FF, Azᶠᶠᵃ); set!(CF, Azᶜᶠᵃ); set!(FC, Azᶠᶜᵃ); set!(CC, Azᶜᶜᵃ); 
    fill_halo_regions!((FF, CF, FC, CC))
    Azᶠᶠᵃ = FF.data[:, :, 1]; 
    Azᶜᶠᵃ = CF.data[:, :, 1]; 
    Azᶠᶜᵃ = FC.data[:, :, 1]; 
    Azᶜᶜᵃ = CC.data[:, :, 1]; 

    Hx, Hy, Hz = halo

    grid = OrthogonalSphericalShellGrid{Periodic, RightConnected, Bounded}(arch,
                    Nx, Ny, Nz,
                    Hx, Hy, Hz,
                    convert(eltype(radius), Lz),
                    on_architecture(arch,  λᶜᶜᵃ), on_architecture(arch,  λᶠᶜᵃ), on_architecture(arch,  λᶜᶠᵃ), on_architecture(arch,  λᶠᶠᵃ),
                    on_architecture(arch,  φᶜᶜᵃ), on_architecture(arch,  φᶠᶜᵃ), on_architecture(arch,  φᶜᶠᵃ), on_architecture(arch,  φᶠᶠᵃ), on_architecture(arch, zᵃᵃᶜ),  on_architecture(arch, zᵃᵃᶠ),
                    on_architecture(arch, Δxᶜᶜᵃ), on_architecture(arch, Δxᶠᶜᵃ), on_architecture(arch, Δxᶜᶠᵃ), on_architecture(arch, Δxᶠᶠᵃ),
                    on_architecture(arch, Δyᶜᶜᵃ), on_architecture(arch, Δyᶜᶠᵃ), on_architecture(arch, Δyᶠᶜᵃ), on_architecture(arch, Δyᶠᶠᵃ), on_architecture(arch, Δzᵃᵃᶜ), on_architecture(arch, Δzᵃᵃᶠ),
                    on_architecture(arch, Azᶜᶜᵃ), on_architecture(arch, Azᶠᶜᵃ), on_architecture(arch, Azᶜᶠᵃ), on_architecture(arch, Azᶠᶠᵃ),
                    radius, Tripolar())
                                                        
    return grid
end

function save_grid_metrics!(filename, grid::TripolarGrid)
    (; λᶜᶜᵃ, φᶜᶜᵃ, λᶜᶠᵃ, φᶜᶠᵃ, λᶠᶜᵃ, φᶠᶜᵃ, λᶠᶠᵃ, φᶠᶠᵃ, Δxᶜᶜᵃ, Δyᶜᶜᵃ, Δxᶜᶠᵃ, Δyᶜᶠᵃ, Δxᶠᶜᵃ, Δyᶠᶜᵃ, Δxᶠᶠᵃ, Δyᶠᶠᵃ, Azᶜᶜᵃ, Azᶜᶠᵃ, Azᶠᶜᵃ, Azᶠᶠᵃ) = grid
    @save filename λᶜᶜᵃ φᶜᶜᵃ λᶜᶠᵃ φᶜᶠᵃ λᶠᶜᵃ φᶠᶜᵃ λᶠᶠᵃ φᶠᶠᵃ Δxᶜᶜᵃ Δyᶜᶜᵃ Δxᶜᶠᵃ Δyᶜᶠᵃ Δxᶠᶜᵃ Δyᶠᶜᵃ Δxᶠᶠᵃ Δyᶠᶠᵃ Azᶜᶜᵃ Azᶜᶠᵃ Azᶠᶜᵃ Azᶠᶠᵃ

    return nothing
end

function TripolarGrid(filename::String;
    arch = CPU(),
    halos = (5, 5, 5),
    Nz, 
    z)

    @load filename λᶜᶜᵃ φᶜᶜᵃ λᶜᶠᵃ φᶜᶠᵃ λᶠᶜᵃ φᶠᶜᵃ λᶠᶠᵃ φᶠᶠᵃ Δxᶜᶜᵃ Δyᶜᶜᵃ Δxᶜᶠᵃ Δyᶜᶠᵃ Δxᶠᶜᵃ Δyᶠᶜᵃ Δxᶠᶠᵃ Δyᶠᶠᵃ Azᶜᶜᵃ Azᶜᶠᵃ Azᶠᶜᵃ Azᶠᶠᵃ

    TX, TY, TZ = (Periodic, RightConnected, Bounded)
    Hx, Hy, Hz = halos

    Nx, Ny = size(λᶜᶜᵃ) .- 2 .* (Hx, Hy)

    z_grid = RectilinearGrid(arch; size = Nz, halo = halos[3], z, topology = (Flat, Flat, Bounded))

    grid = OrthogonalSphericalShellGrid{TX, TY, TZ}(CPU(),
                                                    Nx, Ny, Nz,
                                                    Hx, Hy, Hz,
                                                    z_grid.Lz,
                                                    λᶜᶜᵃ,  λᶠᶜᵃ,  λᶜᶠᵃ,  λᶠᶠᵃ,
                                                    φᶜᶜᵃ,  φᶠᶜᵃ,  φᶜᶠᵃ,  φᶠᶠᵃ, z_grid.zᵃᵃᶜ, z_grid.zᵃᵃᶠ,
                                                    Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                                                    Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶜᵃ, Δyᶠᶠᵃ, z_grid.Δzᵃᵃᶜ, z_grid.Δzᵃᵃᶠ,
                                                    Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ,
                                                    R_Earth,
                                                    TripolarMapping())

    return on_architecture(arch, grid)
end

import Oceananigans.Grids: x_domain, y_domain

x_domain(grid::TripolarGrid) = CUDA.@allowscalar 0, 360
y_domain(grid::TripolarGrid) = CUDA.@allowscalar minimum(grid.φᶠᶠᵃ), 90

import Oceananigans.BoundaryConditions: regularize_field_boundary_conditions
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, 
                                       assumed_field_location, 
                                       regularize_boundary_condition

function regularize_field_boundary_conditions(bcs::FieldBoundaryConditions,
                                              grid::TripolarGrid,
                                              field_name::Symbol,
                                              prognostic_names=nothing)

    loc = assumed_field_location(field_name)

    sign = (field_name == :u || field_name == :v) ? -1 : 1

    west   = regularize_boundary_condition(bcs.west,   grid, loc, 1, LeftBoundary,  prognostic_names)
    east   = regularize_boundary_condition(bcs.east,   grid, loc, 1, RightBoundary, prognostic_names)
    south  = regularize_boundary_condition(bcs.south,  grid, loc, 2, LeftBoundary,  prognostic_names)
    north  = ZipperBoundaryCondition(sign)
    bottom = regularize_boundary_condition(bcs.bottom, grid, loc, 3, LeftBoundary,  prognostic_names)
    top    = regularize_boundary_condition(bcs.top,    grid, loc, 3, RightBoundary, prognostic_names)

    immersed = regularize_immersed_boundary_condition(bcs.immersed, grid, loc, field_name, prognostic_names)

    return FieldBoundaryConditions(west, east, south, north, bottom, top, immersed)
end

import Oceananigans.Fields: Field
using Oceananigans.Fields: architecture, 
                           validate_indices, 
                           validate_boundary_conditions, 
                           FieldBoundaryConditions, 
                           FieldBoundaryBuffers


sign(::Face, ::Face)     = -1
sign(::Face, ::Center)   = -1
sign(::Center, ::Face)   = -1
sign(::Center, ::Center) = 1

function Field((LX, LY, LZ)::Tuple, grid::TripolarGrid, data, old_bcs, indices::Tuple, op, status)
    arch = architecture(grid)
    indices = validate_indices(indices, (LX, LY, LZ), grid)
    validate_field_data((LX, LY, LZ), data, grid, indices)
    validate_boundary_conditions((LX, LY, LZ), grid, old_bcs)
    new_bcs = FieldBoundaryConditions(; west = old_bcs.west, 
                                        east = old_bcs.east, 
                                        south = old_bcs.south,
                                        north = ZipperBoundaryCondition(sign(LX, LY)),
                                        top = old_bcs.top,
                                        bottom = old_bcs.bottom)

    buffers = FieldBoundaryBuffers(grid, data, new_bcs)

    return Field{LX, LY, LZ}(grid, data, new_bcs, indices, op, status, buffers)
end
