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
@kernel function compute_tripolar_coords!(jnum, xnum, ynum, Δλᶠᵃᵃ, Δj, Jeq, Nφ, a_interpolator, b_interpolator, c_interpolator)
    i = @index(Global, Linear)
    N = size(xnum, 2)
    @inbounds begin
        h = (90 - Δλᶠᵃᵃ * i) * 2π / 360
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

@inline tripolar_stretching_function(φ) = (φ / 145)^2

@inline cosine_a_curve(φ) = - equator_fcurve(φ) 
@inline cosine_b_curve(φ) = - equator_fcurve(φ) + ifelse(φ > 0, tripolar_stretching_function(φ), 0)

@inline zero_c_curve(φ) = 0

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
                      halo        = (4, 4, 4), 
                      radius      = R_Earth, 
                      z           = (0, 1),
                      singularity_longitude = 230,
                      Nproc       = 10000,
                      Nnum        = 10000,
                      a_curve     = cosine_a_curve,
                      b_curve     = cosine_b_curve,
                      c_curve     = zero_c_curve)

    # For now, only for domains Periodic in λ (from -180 to 180 degrees) and Bounded in φ.
    # φ has to reach the north pole.`
    # For all the rest we can easily use a `LatitudeLongitudeGrid` without warping

    latitude  = (southermost_latitude, 90)
    longitude = (-180, 180) 
    
    Nλ, Nφ, Nz = size
    Hλ, Hφ, Hz = halo

    Lφ, φᵃᶠᵃ, φᵃᶜᵃ, Δφᵃᶠᵃ, Δφᵃᶜᵃ = generate_coordinate(FT, Bounded(),  Nφ, Hφ, latitude,  :φ, CPU())
    Lλ, λᶠᵃᵃ, λᶜᵃᵃ, Δλᶠᵃᵃ, Δλᶜᵃᵃ = generate_coordinate(FT, Periodic(), Nλ, Hλ, longitude, :λ, CPU())
    Lz, zᵃᵃᶠ, zᵃᵃᶜ, Δzᵃᵃᶠ, Δzᵃᵃᶜ = generate_coordinate(FT, Bounded(),  Nz, Hz, z,         :z, CPU())

    λF = zeros(Nλ+1, Nφ+1)
    φF = zeros(Nλ+1, Nφ+1)

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

    x = zeros(Nλ+1, 1:Nφ+1)
    y = zeros(Nλ+1, 1:Nφ+1)

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

    Nnum = 10000
    xnum = zeros(1:Nλ+1, Nnum)
    ynum = zeros(1:Nλ+1, Nnum)
    jnum = zeros(1:Nλ+1, Nnum)

    loop! = compute_tripolar_coords!(device(CPU()), min(256, Nλ+1), Nλ+1)
    loop!(jnum, xnum, ynum, Δλᶠᵃᵃ, 1/Nnum, Jeq, Nφ, a_interpolator, b_interpolator, c_interpolator) 

    loop! = _compute_coordinates!(device(CPU()), (16, 16), (Nλ+1, Nφ+1))
    loop!(λF, φF, x, y, Jeq, Δλᶠᵃᵃ, φᵃᶠᵃ, a_curve, xnum, ynum, jnum)
    
    # Rotate the λ direction accordingly
    for i in 1:Nλ÷2
        λF[i, :] .-= 90
        λF[i+Nλ÷2, :] .+= 90
    end 

    λF = circshift(λF, (1, 0))
    φF = circshift(φF, (1, 0))
        
    Nx = Base.size(λF, 1)
    Ny = Base.size(λF, 2) 

    # Helper grid to fill halo metrics
    grid = RectilinearGrid(; size = (Nx, Ny, 1), halo, topology = (Periodic, RightConnected, Bounded), z = (0, 1), x = (0, 1), y = (0, 1))

    default_boundary_conditions = FieldBoundaryConditions(north  = ZipperBoundaryCondition(),
                                                          south  = FluxBoundaryCondition(nothing),
                                                          west   = Oceananigans.PeriodicBoundaryCondition(),
                                                          east   = Oceananigans.PeriodicBoundaryCondition(),
                                                          top    = nothing,
                                                          bottom = nothing)
                                                          
    lF = Field((Face, Face, Center), grid; boundary_conditions = default_boundary_conditions)
    pF = Field((Face, Face, Center), grid; boundary_conditions = default_boundary_conditions)

    @show Base.size(lF), Base.size(λF)
    set!(lF, λF)
    set!(pF, φF)

    fill_halo_regions!((lF, pF))

    λᶠᶠᵃ = lF.data[:, :, 1]
    φᶠᶠᵃ = pF.data[:, :, 1]

    λᶠᶠᵃ[:, 0] .= λᶠᶠᵃ[:, 1]
    φᶠᶠᵃ[:, 0] .= φᶠᶠᵃ[:, 1]

    λᶠᶠᵃ[:, Ny+1] .= λᶠᶠᵃ[:, Ny]
    φᶠᶠᵃ[:, Ny+1] .= φᶠᶠᵃ[:, Ny]

    λᶜᶠᵃ = OffsetArray(zeros(Base.size(λᶠᶠᵃ)), λᶠᶠᵃ.offsets...)
    λᶜᶜᵃ = OffsetArray(zeros(Base.size(λᶠᶠᵃ)), λᶠᶠᵃ.offsets...)

    λᶠᶜᵃ = 0.5 .* OffsetArray(λᶠᶠᵃ.parent[:, 2:end] .+ λᶠᶠᵃ.parent[:, 1:end-1], λᶠᶠᵃ.offsets...);
    φᶠᶜᵃ = 0.5 .* OffsetArray(φᶠᶠᵃ.parent[:, 2:end] .+ φᶠᶠᵃ.parent[:, 1:end-1], φᶠᶠᵃ.offsets...);
    φᶜᶠᵃ = 0.5 .* OffsetArray(φᶠᶠᵃ.parent[2:end, :] .+ φᶠᶠᵃ.parent[1:end-1, :], φᶠᶠᵃ.offsets...);
    φᶜᶜᵃ = 0.5 .* OffsetArray(φᶜᶠᵃ.parent[:, 2:end] .+ φᶜᶠᵃ.parent[:, 1:end-1], φᶜᶠᵃ.offsets...);

    # The λᶜᶠᵃ points need to be handled individually (λ jumps between -180 and 180)
    # and cannot average between them, find a better way to do this
    for i in 1:Base.size(λᶜᶠᵃ, 1) - 1
        for j in 1:Base.size(λᶜᶠᵃ, 2) - 1
            λᶜᶠᵃ.parent[i, j] = if abs(λᶠᶠᵃ.parent[i+1, j] .- λᶠᶠᵃ.parent[i, j]) > 100
                (λᶠᶠᵃ.parent[i+1, j] .- λᶠᶠᵃ.parent[i, j]) / 2
            else
                (λᶠᶠᵃ.parent[i+1, j] .+ λᶠᶠᵃ.parent[i, j]) / 2
            end
        end
    end

    λᶜᶜᵃ = 0.5 .* OffsetArray(λᶜᶠᵃ.parent[:, 2:end] .+ λᶜᶠᵃ.parent[:, 1:end-1], λᶜᶠᵃ.offsets...);

    for λ in (λᶜᶠᵃ, λᶠᶜᵃ, λᶠᶠᵃ, λᶜᶜᵃ)
        λ .+= singularity_longitude
        λ .=  convert_to_0_360.(λ)
    end

    # Metrics
    Δxᶜᶜᵃ = zeros(Nx, Ny  )
    Δxᶠᶜᵃ = zeros(Nx, Ny  )
    Δxᶜᶠᵃ = zeros(Nx, Ny+1)
    Δxᶠᶠᵃ = zeros(Nx, Ny+1)

    Δyᶜᶜᵃ = zeros(Nx, Ny  )
    Δyᶠᶜᵃ = zeros(Nx, Ny  )
    Δyᶜᶠᵃ = zeros(Nx, Ny+1)
    Δyᶠᶠᵃ = zeros(Nx, Ny+1)

    Azᶜᶜᵃ = zeros(Nx, Ny  )
    Azᶠᶜᵃ = zeros(Nx, Ny  )
    Azᶜᶠᵃ = zeros(Nx, Ny+1)
    Azᶠᶠᵃ = zeros(Nx, Ny+1)

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
                    arch_array(arch,  λᶜᶜᵃ), arch_array(arch,  λᶠᶜᵃ), arch_array(arch,  λᶜᶠᵃ), arch_array(arch,  λᶠᶠᵃ),
                    arch_array(arch,  φᶜᶜᵃ), arch_array(arch,  φᶠᶜᵃ), arch_array(arch,  φᶜᶠᵃ), arch_array(arch,  φᶠᶠᵃ), arch_array(arch, zᵃᵃᶜ),  arch_array(arch, zᵃᵃᶠ),
                    arch_array(arch, Δxᶜᶜᵃ), arch_array(arch, Δxᶠᶜᵃ), arch_array(arch, Δxᶜᶠᵃ), arch_array(arch, Δxᶠᶠᵃ),
                    arch_array(arch, Δyᶜᶜᵃ), arch_array(arch, Δyᶜᶠᵃ), arch_array(arch, Δyᶠᶜᵃ), arch_array(arch, Δyᶠᶠᵃ), arch_array(arch, Δzᵃᵃᶜ), arch_array(arch, Δzᵃᵃᶠ),
                    arch_array(arch, Azᶜᶜᵃ), arch_array(arch, Azᶠᶜᵃ), arch_array(arch, Azᶜᶠᵃ), arch_array(arch, Azᶠᶠᵃ),
                    radius, Tripolar())
                                                        
    return grid
end
