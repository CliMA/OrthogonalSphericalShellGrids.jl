"""
    struct WarpedLatitudeLongitude

A struct representing a warped latitude-longitude grid.

TODO: put here information about the grid, i.e.: 

1) north pole latitude and longitude
2) functions used to construct the Grid
3) Numerical discretization used to construct the Grid
4) Last great circle size in degrees
"""
struct WarpedLatitudeLongitude end

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
@kernel function compute_coords!(jnum, xnum, ynum, Δλᶠᵃᵃ, Jeq, Nφ, f_interpolator, g_interpolator)
    i = @index(Global, Linear)
    N = size(xnum, 2)
    @inbounds begin
        h = (90 - Δλᶠᵃᵃ * i) * 2π / 360
        xnum[i, 1], ynum[i, 1] = cos(h), sin(h)
        Δx = xnum[i, 1] / N
        xnum[i, 2] = xnum[i, 1] - Δx
        ynum[i, 2] = ynum[i, 1] - Δx * tan(h)
        for n in 3:N
            # Great circles
            func(x) = xnum[i, n-1]^2 + ynum[i, n-1]^2 - ynum[i, n-1] * (f_interpolator(x) + g_interpolator(x)) + f_interpolator(x) * g_interpolator(x)
            jnum[i, n-1] = secant_root_find(Jeq, Jeq+1, func, Nφ+1)
            xnum[i, n]   = xnum[i, n-1] - Δx
            # Adams-Bashforth-2 integrator to find the perpendicular to the circle
            ynum[i, n]   = ynum[i, n-1] - Δx * (1.5 * (2ynum[i, n-1] - f_interpolator(jnum[i, n-1]) - g_interpolator(jnum[i, n-1])) / (2 * xnum[i, n-1]) - 
                                                0.5 * (2ynum[i, n-2] - f_interpolator(jnum[i, n-2]) - g_interpolator(jnum[i, n-1])) / (2 * xnum[i, n-2]))
        end
        @show i
    end
end

@inline stretching_function(φ) = (φ^2 / 145^2)

@inline equator_fcurve(φ)      = - sqrt((tan((90 - φ) / 360 * π))^2)
@inline quadratic_f_curve(φ)   =   equator_fcurve(φ) + ifelse(φ > 0, stretching_function(φ), 0)
@inline quadratic_g_curve(φ)   = - equator_fcurve(φ) + ifelse(φ > 0, stretching_function(φ), 0)

"""
    WarpedLatitudeLongitudeGrid(arch = CPU(), FT::DataType = Float64; 
                                size, 
                                southermost_latitude = -75, 
                                halo        = (4, 4, 4), 
                                radius      = R_Earth, 
                                z           = (0, 1),
                                singularity_longitude = 230,
                                f_curve     = quadratic_f_curve,
                                g_curve     = quadratic_g_curve)

Constructs a warped latitude-longitude grid on a spherical shell.

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

A `OrthogonalSphericalShellGrid` object representing the warped latitude-longitude grid.
"""
function WarpedLatitudeLongitudeGrid(arch = CPU(), FT::DataType = Float64; 
                                     size, 
                                     southermost_latitude = -85, 
                                     halo        = (4, 4, 4), 
                                     radius      = R_Earth, 
                                     z           = (0, 1),
                                     singularity_longitude = 230,
                                     f_curve     = quadratic_f_curve,
                                     g_curve     = quadratic_g_curve)

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

    fⱼ = zeros(1:Nφ+1)
    gⱼ = zeros(1:Nφ+1)

    x = zeros(Nλ+1, 1:Nφ+1)
    y = zeros(Nλ+1, 1:Nφ+1)

    # Shif pole upwards
    for j in 1:Nφ+1
        fⱼ[j] = f_curve(φᵃᶠᵃ[j])
        gⱼ[j] = g_curve(φᵃᶠᵃ[j]) 
    end

    fy = fⱼ
    gy = gⱼ
    fx = Float64.(collect(1:Nφ+1))

    f_interpolator(j) = linear_interpolate(j, fx, fy)
    g_interpolator(j) = linear_interpolate(j, fx, gy)

    @info "I am here!"

    Nsol = 1000
    xnum = zeros(1:Nλ+1, Nsol)
    ynum = zeros(1:Nλ+1, Nsol)
    jnum = zeros(1:Nλ+1, Nsol)

    loop! = compute_coords!(device(CPU()), min(256, Nλ+1), Nλ+1)
    loop!(jnum, xnum, ynum, Δλᶠᵃᵃ, Jeq, Nφ, f_interpolator, g_interpolator) 

    loop! = _compute_coordinates!(device(CPU()), (16, 16), (Nλ+1, Nφ+1))
    loop!(λF, φF, x, y, Jeq, Δλᶠᵃᵃ, φᵃᶠᵃ, f_curve, xnum, ynum, jnum)

    # Rotate the λ direction accordingly
    for i in 1:Nλ÷2
        λF[i, :] .-= 90
        λF[i+Nλ÷2, :] .+= 90
    end 

    # Remove the top of the grid for now 10, then figure out a way
    # to give a choice for the last great circle size
    λF = λF[1:end-1, 1:end-10]
    φF = φF[1:end-1, 1:end-10]

    λF = circshift(λF, (1, 0))
    φF = circshift(φF, (1, 0))
        
    Nx = Base.size(λF, 1)
    Ny = Base.size(λF, 2) - 1

    # Helper grid to fill halo metrics
    grid = RectilinearGrid(; size = (Nx, Ny, 1), halo, topology = (Periodic, Bounded, Bounded), z = (0, 1), x = (0, 1), y = (0, 1))

    lF = Field((Face, Face, Center), grid)
    pF = Field((Face, Face, Center), grid)

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
    FF = Field((Face, Face, Center),     grid)
    FC = Field((Face, Center, Center),   grid)
    CF = Field((Center, Face, Center),   grid)
    CC = Field((Center, Center, Center), grid)

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

    grid = OrthogonalSphericalShellGrid{Periodic, Bounded, Bounded}(arch,
                    Nx, Ny, Nz,
                    Hx, Hy, Hz,
                    convert(eltype(radius), Lz),
                    on_architecture(arch,  λᶜᶜᵃ), on_architecture(arch,  λᶠᶜᵃ), on_architecture(arch,  λᶜᶠᵃ), on_architecture(arch,  λᶠᶠᵃ),
                    on_architecture(arch,  φᶜᶜᵃ), on_architecture(arch,  φᶠᶜᵃ), on_architecture(arch,  φᶜᶠᵃ), on_architecture(arch,  φᶠᶠᵃ), on_architecture(arch, zᵃᵃᶜ),  on_architecture(arch, zᵃᵃᶠ),
                    on_architecture(arch, Δxᶜᶜᵃ), on_architecture(arch, Δxᶠᶜᵃ), on_architecture(arch, Δxᶜᶠᵃ), on_architecture(arch, Δxᶠᶠᵃ),
                    on_architecture(arch, Δyᶜᶜᵃ), on_architecture(arch, Δyᶜᶠᵃ), on_architecture(arch, Δyᶠᶜᵃ), on_architecture(arch, Δyᶠᶠᵃ), on_architecture(arch, Δzᵃᵃᶜ), on_architecture(arch, Δzᵃᵃᶠ),
                    on_architecture(arch, Azᶜᶜᵃ), on_architecture(arch, Azᶠᶜᵃ), on_architecture(arch, Azᶜᶠᵃ), on_architecture(arch, Azᶠᶠᵃ),
                    radius, WarpedLatitudeLongitude())
                                                        
    return grid
end
