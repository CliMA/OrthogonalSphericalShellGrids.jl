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
@kernel function compute_coords!(jnum, xnum, ynum, λ₀, Δλ, Jeq, Nφ, f_interpolator, g_interpolator)
    i = @index(Global, Linear)
    N = size(xnum, 2)
    @inbounds begin
        h = (λ₀ - Δλ * i) * 2π / 360
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

const WarpedLatitudeLongitudeGrid{FT, TX, TY, TZ, A, R, FR, Arch} = OrthogonalSphericalShellGrid{FT, TX, TY, TZ, A, R, FR, <:WarpedLatitudeLongitude, Arch}

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
                                     Nnum        = 1000,
                                     Nproc       = 1000,
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

    λFF = zeros(Nλ, Nφ+1)
    φFF = zeros(Nλ, Nφ+1)
    λFC = zeros(Nλ, Nφ)
    φFC = zeros(Nλ, Nφ)

    λCF = zeros(Nλ, Nφ+1)
    φCF = zeros(Nλ, Nφ+1)
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

    fᶠⱼ = zeros(Nproc+1)
    gᶠⱼ = zeros(Nproc+1)
    
    fᶜⱼ = zeros(Nproc+1)
    gᶜⱼ = zeros(Nproc+1)
    
    xnum = zeros(Nλ+1, Nnum)
    ynum = zeros(Nλ+1, Nnum)
    jnum = zeros(Nλ+1, Nnum)

    φproc = range(φᵃᶠᵃ[1], φᵃᶠᵃ[Nφ+1], length = Nproc + 1) 
    
    # calculate the eccentricities of the ellipse
    for (j, φ) in enumerate(φproc)
        fᶠⱼ[j] = f_curve(φ)
        gᶠⱼ[j] = g_curve(φ) 
    end

    fᶠx = Float64.(collect(0:Nproc) ./ Nproc * Nφ .+ 1)
    
    f_face_interp(j) = linear_interpolate(j, fᶠx, fᶠⱼ)
    g_face_interp(j) = linear_interpolate(j, fᶠx, gᶠⱼ)
    
    φproc = range(φᵃᶜᵃ[1], φᵃᶜᵃ[Nφ+1], length = Nproc + 1) 
    
    # calculate the eccentricities of the ellipse
    for (j, φ) in enumerate(φproc)
        fᶜⱼ[j] = f_curve(φ)
        gᶜⱼ[j] = g_curve(φ) 
    end

    f_center_interp(j) = linear_interpolate(j, fᶠx, fᶜⱼ)
    g_center_interp(j) = linear_interpolate(j, fᶠx, gᶜⱼ)

    xnum = zeros(Nλ+1, Nnum)
    ynum = zeros(Nλ+1, Nnum)
    jnum = zeros(Nλ+1, Nnum)

    # X - Face coordinates
    λ₀ = 90 # ᵒ degrees  

    loop! = compute_coords!(device(CPU()), min(256, Nλ+1), Nλ+1)
    loop!(jnum, xnum, ynum, λ₀, Δλᶠᵃᵃ, Jeq, Nφ, f_face_interp, g_face_interp)

    loop! = _compute_coordinates!(device(CPU()), (16, 16), (Nλ, Nφ+1))
    loop!(λFF, φFF, Jeq, λ₀, Δλᶠᵃᵃ, φᵃᶠᵃ, f_curve, xnum, ynum, jnum, Nλ)
    
    loop! = compute_coords!(device(CPU()), min(256, Nλ+1), Nλ+1)
    loop!(jnum, xnum, ynum, λ₀, Δλᶠᵃᵃ, Jeq, Nφ, f_center_interp, g_center_interp)

    loop! = _compute_coordinates!(device(CPU()), (16, 16), (Nλ, Nφ))
    loop!(λFC, φFC, Jeq, λ₀, Δλᶠᵃᵃ, φᵃᶜᵃ, f_curve, xnum, ynum, jnum, Nλ)

    # X - Face coordinates
    λ₀ = 90 + Δλᶜᵃᵃ / 2 # ᵒ degrees  

    loop! = compute_coords!(device(CPU()), min(256, Nλ+1), Nλ+1)
    loop!(jnum, xnum, ynum, λ₀, Δλᶜᵃᵃ, Jeq, Nφ, f_face_interp, g_face_interp)

    loop! = _compute_coordinates!(device(CPU()), (16, 16), (Nλ, Nφ+1))
    loop!(λCF, φCF, Jeq, λ₀, Δλᶜᵃᵃ, φᵃᶠᵃ, f_curve, xnum, ynum, jnum, Nλ)
    
    loop! = compute_coords!(device(CPU()), min(256, Nλ+1), Nλ+1)
    loop!(jnum, xnum, ynum, λ₀, Δλᶜᵃᵃ, Jeq, Nφ, f_center_interp, g_center_interp)

    loop! = _compute_coordinates!(device(CPU()), (16, 16), (Nλ, Nφ))
    loop!(λCC, φCC, Jeq, λ₀, Δλᶜᵃᵃ, φᵃᶜᵃ, f_curve, xnum, ynum, jnum, Nλ)

    Nx = Nλ
    Ny = Nφ

    # Helper grid to fill halo metrics
    grid = RectilinearGrid(; size = (Nx, Ny, 1), halo, topology = (Periodic, Bounded, Bounded), z = (0, 1), x = (0, 1), y = (0, 1))
                            
    lFF = Field((Face, Face, Center), grid)
    pFF = Field((Face, Face, Center), grid)

    lFC = Field((Face, Center, Center), grid)
    pFC = Field((Face, Center, Center), grid)
    
    lCF = Field((Center, Face, Center), grid)
    pCF = Field((Center, Face, Center), grid)

    lCC = Field((Center, Center, Center), grid)
    pCC = Field((Center, Center, Center), grid)

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

const WRG = Union{WarpedLatitudeLongitudeGrid, ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:WarpedLatitudeLongitudeGrid}}
