module OrthogonalSphericalShellGrids

export WarpedLatitudeLongitudeGrid

using Oceananigans
using Oceananigans.Grids: R_Earth

using Oceananigans.Fields: index_binary_search
using Oceananigans.Architectures: device, arch_array
using JLD2

using Oceananigans.Grids: halo_size, spherical_area_quadrilateral
using Oceananigans.Grids: lat_lon_to_cartesian
using OffsetArrays
using Oceananigans.Operators

using KernelAbstractions: @kernel, @index
using Oceananigans.Grids: generate_coordinate

using Oceananigans.BoundaryConditions

# Put here information about the grid, i.e.: 
# 1) north pole latitude and longitude
# 2) functions used to construct the Grid
# 3) Numerical discretization used to construct the Grid
# 4) Last great circle size in degrees
struct WarpedLatitudeLongitude end

@inline function linear_interpolate(x₀, x, y, offset) 
    x₁, x₂ = index_binary_search(x, x₀, length(x))

    x₁ += ceil(Int, offset)
    x₂ += ceil(Int, offset)

    @inbounds begin
        y₁ = y[x₁]
        y₂ = y[x₂]
    end

    if x₁ == x₂
        return y₁
    else
        return (y₂ - y₁) / (x₂ - x₁) * (x₀ - x₁) + y₁
    end
end

@inline function linear_interpolate(x₀, x, y) 
    i₁, i₂ = index_binary_search(x, x₀, length(x))

    @inbounds begin
        x₁ = x[i₁]
        x₂ = x[i₂]
        y₁ = y[i₁]
        y₂ = y[i₂]
    end

    if x₁ == x₂
        return y₁
    else
        return (y₂ - y₁) / (x₂ - x₁) * (x₀ - x₁) + y₁
    end
end

function secant_root_find(j₀, j₁, f; tol = 1e-12)
    r = j₁ - f(j₁) * (j₁ - j₀) / (f(j₁) - f(j₀)) 
    while abs(f(r)) > tol
        j₀ = j₁
        j₁ = r
        r = j₁ - f(j₁) * (j₁ - j₀) / (f(j₁) - f(j₀)) 
    end
    return r
end

@kernel function compute_coords!(jnum, xnum, ynum, Δλᶠᵃᵃ, Jeq, f_interpolator, g_interpolator)
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
            jnum[i, n-1] = secant_root_find(Jeq, Jeq+1, func)
            xnum[i, n]   = xnum[i, n-1] - Δx
            # Adams-Bashforth-2 integrator to find the perpendicular to the circle
            ynum[i, n]   = ynum[i, n-1] - Δx * (1.5 * (2ynum[i, n-1] - f_interpolator(jnum[i, n-1]) - g_interpolator(jnum[i, n-1])) / (2 * xnum[i, n-1]) - 
                                                0.5 * (2ynum[i, n-2] - f_interpolator(jnum[i, n-2]) - g_interpolator(jnum[i, n-1])) / (2 * xnum[i, n-2]))
        end
        @show i
    end
end

function haversine(a, b, radius)
    λ₁, φ₁ = a
    λ₂, φ₂ = b

    x₁, y₁, z₁ = lat_lon_to_cartesian(φ₁, λ₁, radius)
    x₂, y₂, z₂ = lat_lon_to_cartesian(φ₂, λ₂, radius)

    return radius * acos(max(-1.0, min((x₁ * x₂ + y₁ * y₂ + z₁ * z₂) / radius^2, 1.0)))
end

@inline equator_fcurve(φ)      = - sqrt((tan((90 - φ) / 360 * π))^2)
@inline stretching_function(φ) = (φ^2 / 145^2)
@inline quadratic_f_curve(φ)   =   equator_fcurve(φ) + ifelse(φ > 0, stretching_function(φ), 0)
@inline quadratic_g_curve(φ)   = - equator_fcurve(φ) + ifelse(φ > 0, stretching_function(φ), 0)

# For now, only for domains Periodic in λ (from -180 to 180 degrees) and Bounded in φ.
# φ has to reach the north pole.
# For all the rest we can easily use a `LatitudeLongitudeGrid` without warping
function WarpedLatitudeLongitudeGrid(arch = CPU(), FT::DataType = Float64; 
                                     size, 
                                     southermost_latitude = -75, 
                                     halo        = (4, 4, 4), 
                                     radius      = R_Earth, 
                                     z           = (0, 1),
                                     f_curve     = quadratic_f_curve,
                                     g_curve     = quadratic_g_curve)

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

    Nsol = 1000
    xnum = zeros(1:Nλ+1, Nsol)
    ynum = zeros(1:Nλ+1, Nsol)
    jnum = zeros(1:Nλ+1, Nsol)

    loop! = compute_coords!(device(CPU()), min(256, Nλ+1), Nλ+1)
    loop!(jnum, xnum, ynum, Δλᶠᵃᵃ, Jeq, f_interpolator, g_interpolator) 

    for i in 1:Nλ+1
        for j in 1:Jeq-1
            h = (90 - Δλᶠᵃᵃ * i) * 2π / 360
            x[i, j] = - fⱼ[j] * cos(h)
            y[i, j] = - fⱼ[j] * sin(h)
        end
        for j in Jeq:Nφ+1
            x[i, j]  = linear_interpolate(j, jnum[i, :], xnum[i, :])
            y[i, j]  = linear_interpolate(j, jnum[i, :], ynum[i, :])
        end
    end
    
    for i in 1:Nλ+1
        for j in 1:Nφ+1
            λF[i, j] = - 180 / π * (atan(y[i, j] / x[i, j]))              
            φF[i, j] = 90 - 360 / π * atan(sqrt(y[i, j]^2 + x[i, j]^2)) 
        end
    end

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

    @inbounds begin
        for i in 1:Nx, j in 1:Ny
            Δxᶜᶜᵃ[i, j] = haversine((λᶠᶜᵃ[i+1, j], φᶠᶜᵃ[i+1, j]), (λᶠᶜᵃ[i, j],   φᶠᶜᵃ[i, j]),   radius)
            Δxᶠᶜᵃ[i, j] = haversine((λᶜᶜᵃ[i, j],   φᶜᶜᵃ[i, j]),   (λᶜᶜᵃ[i-1, j], φᶜᶜᵃ[i-1, j]), radius)
            Δxᶜᶠᵃ[i, j] = haversine((λᶠᶠᵃ[i+1, j], φᶠᶠᵃ[i+1, j]), (λᶠᶠᵃ[i, j],   φᶠᶠᵃ[i, j]),   radius)
            Δxᶠᶠᵃ[i, j] = haversine((λᶜᶠᵃ[i, j],   φᶜᶠᵃ[i, j]),   (λᶜᶠᵃ[i-1, j], φᶜᶠᵃ[i-1, j]), radius)

            Δyᶜᶜᵃ[i, j] = haversine((λᶜᶠᵃ[i, j+1], φᶜᶠᵃ[i, j+1]),   (λᶜᶠᵃ[i, j],   φᶜᶠᵃ[i, j]),   radius)
            Δyᶜᶠᵃ[i, j] = haversine((λᶜᶜᵃ[i, j  ],   φᶜᶜᵃ[i, j]),   (λᶜᶜᵃ[i, j-1], φᶜᶜᵃ[i, j-1]), radius)
            Δyᶠᶜᵃ[i, j] = haversine((λᶠᶠᵃ[i, j+1], φᶠᶠᵃ[i, j+1]),   (λᶠᶠᵃ[i, j],   φᶠᶠᵃ[i, j]),   radius)
            Δyᶠᶠᵃ[i, j] = haversine((λᶠᶜᵃ[i, j  ],   φᶠᶜᵃ[i, j]),   (λᶠᶜᵃ[i, j-1], φᶠᶜᵃ[i, j-1]), radius)
        
            a = lat_lon_to_cartesian(φᶠᶠᵃ[ i ,  j ], λᶠᶠᵃ[ i ,  j ], 1)
            b = lat_lon_to_cartesian(φᶠᶠᵃ[i+1,  j ], λᶠᶠᵃ[i+1,  j ], 1)
            c = lat_lon_to_cartesian(φᶠᶠᵃ[i+1, j+1], λᶠᶠᵃ[i+1, j+1], 1)
            d = lat_lon_to_cartesian(φᶠᶠᵃ[ i , j+1], λᶠᶠᵃ[ i , j+1], 1)

            Azᶜᶜᵃ[i, j] = spherical_area_quadrilateral(a, b, c, d) * radius^2

            a = lat_lon_to_cartesian(φᶜᶠᵃ[i-1,  j ], λᶜᶠᵃ[i-1,  j ], 1)
            b = lat_lon_to_cartesian(φᶜᶠᵃ[ i ,  j ], λᶜᶠᵃ[ i ,  j ], 1)
            c = lat_lon_to_cartesian(φᶜᶠᵃ[ i , j+1], λᶜᶠᵃ[ i , j+1], 1)
            d = lat_lon_to_cartesian(φᶜᶠᵃ[i-1, j+1], λᶜᶠᵃ[i-1, j+1], 1)

            Azᶠᶜᵃ[i, j] = spherical_area_quadrilateral(a, b, c, d) * radius^2 

            a = lat_lon_to_cartesian(φᶠᶜᵃ[ i , j-1], λᶠᶜᵃ[ i , j-1], 1)
            b = lat_lon_to_cartesian(φᶠᶜᵃ[i+1, j-1], λᶠᶜᵃ[i+1, j-1], 1)
            c = lat_lon_to_cartesian(φᶠᶜᵃ[i+1,  j ], λᶠᶜᵃ[i+1,  j ], 1)
            d = lat_lon_to_cartesian(φᶠᶜᵃ[ i ,  j ], λᶠᶜᵃ[ i ,  j ], 1)

            Azᶜᶠᵃ[i, j] = spherical_area_quadrilateral(a, b, c, d) * radius^2 

            a = lat_lon_to_cartesian(φᶜᶜᵃ[i-1, j-1], λᶜᶜᵃ[i-1, j-1], 1)
            b = lat_lon_to_cartesian(φᶜᶜᵃ[ i , j-1], λᶜᶜᵃ[ i , j-1], 1)
            c = lat_lon_to_cartesian(φᶜᶜᵃ[ i ,  j ], λᶜᶜᵃ[ i ,  j ], 1)
            d = lat_lon_to_cartesian(φᶜᶜᵃ[i-1,  j ], λᶜᶜᵃ[i-1,  j ], 1)

            Azᶠᶠᵃ[i, j] = spherical_area_quadrilateral(a, b, c, d) * radius^2 
        end
    end

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
                    arch_array(arch,  λᶜᶜᵃ), arch_array(arch,  λᶠᶜᵃ), arch_array(arch,  λᶜᶠᵃ), arch_array(arch,  λᶠᶠᵃ),
                    arch_array(arch,  φᶜᶜᵃ), arch_array(arch,  φᶠᶜᵃ), arch_array(arch,  φᶜᶠᵃ), arch_array(arch,  φᶠᶠᵃ), arch_array(arch, zᵃᵃᶜ),  arch_array(arch, zᵃᵃᶠ),
                    arch_array(arch, Δxᶜᶜᵃ), arch_array(arch, Δxᶠᶜᵃ), arch_array(arch, Δxᶜᶠᵃ), arch_array(arch, Δxᶠᶠᵃ),
                    arch_array(arch, Δyᶜᶜᵃ), arch_array(arch, Δyᶜᶠᵃ), arch_array(arch, Δyᶠᶜᵃ), arch_array(arch, Δyᶠᶠᵃ), arch_array(arch, Δzᵃᵃᶜ), arch_array(arch, Δzᵃᵃᶠ),
                    arch_array(arch, Azᶜᶜᵃ), arch_array(arch, Azᶠᶜᵃ), arch_array(arch, Azᶜᶠᵃ), arch_array(arch, Azᶠᶠᵃ),
                    radius, WarpedLatitudeLongitude())
                                                        
    return grid
end

end
