
@kernel function _compute_tripolar_coordinates!(λ, φ, Jeq, λ₀, Δλ, φ_grid, f_curve, xnum, ynum, jnum, Nλ, loc)
    i, j = @index(Global, NTuple)

    @inbounds begin
        if j < Jeq + 1
            h = (λ₀ - Δλ * i) * 2π / 360
            x = - f_curve(φ_grid[j]) * cos(h)
            y = - f_curve(φ_grid[j]) * sin(h)
        elseif loc == Center() && j == size(φ, 2)
            x = 0
            y = linear_interpolate(j, jnum[i, :], ynum[i, :])
        else
            x = linear_interpolate(j, jnum[i, :], xnum[i, :])
            y = linear_interpolate(j, jnum[i, :], ynum[i, :])
        end
        
        λ[i, j] = - 180 / π * ifelse(x == 0, ifelse(y == 0, 0, atan(Base.sign(y) * Inf)), atan(y / x))
        φ[i, j] = 90 - 360 / π * atan(sqrt(y^2 + x^2)) 
        λ[i, j] += ifelse(i ≤ Nλ÷2, -90, 90)
    end
end

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
@kernel function compute_tripolar_coords!(jnum, xnum, ynum, λᵢ, Δλ, Δj, Jeq, Nφ, a_interpolator, b_interpolator)
    i = @index(Global, Linear)
    N = size(xnum, 2)
    @inbounds begin
        h = (λᵢ - Δλ * i) * 2π / 360
        xnum[i, 1] = - a_interpolator(Jeq) * cos(h) 
        ynum[i, 1] = - a_interpolator(Jeq) * sin(h)  # Starting always from a circumpherence at the equator
        
        Δx = xnum[i, 1] / N

        myfunc(x) = (xnum[i, 1] / a_interpolator(x)) ^2 + (ynum[i, 1] / b_interpolator(x))^2 - 1 
        jnum[i, 1] = bisection_root_find(myfunc, Jeq-1.0, Nφ+1-Δj, Δj)
        
        for n in 2:N
            # Great circles

            func(j) = (xnum[i, n-1] / a_interpolator(j)) ^2 + (ynum[i, n-1] / b_interpolator(j))^2 - 1 
            jnum[i, n-1] = bisection_root_find(func, Jeq-1.0, Nφ+1.0, Δj)

            # Semi-Implicit integrator
            G¹ = a_interpolator(jnum[i, n-1])^2 / b_interpolator(jnum[i, n-1])^2 / xnum[i, n-1]
            
            ynum[i, n] = ynum[i, n-1] / (1 + G¹ * Δx) 
            xnum[i, n] = xnum[i, n-1] - Δx
        end

        @show i
    end
end

"""
    generate_tripolar_metrics!(λFF, φFF, λFC, φFC, λCF, φCF, λCC;
                              size, halo, latitude, longitude,
                              Nproc, Nnum)

Generate tripolar metrics for a spherical shell grid.

# Arguments
- `λFF`: Output array for face-face longitude coordinates.
- `φFF`: Output array for face-face latitude coordinates.
- `λFC`: Output array for face-center longitude coordinates.
- `φFC`: Output array for face-center latitude coordinates.
- `λCF`: Output array for center-face longitude coordinates.
- `φCF`: Output array for center-face latitude coordinates.
- `λCC`: Output array for center-center longitude coordinates.
- `φCC`: Output array for center-center latitude coordinates.
- `size`: Size of the grid.
- `halo`: Halo size.
- `latitude`: Latitude of the grid.
- `longitude`: Longitude of the grid.
- `Nproc`: Number of processors.
- `Nnum`: Number of numerical points.

# Returns
- `nothing`
"""
function generate_tripolar_metrics!(Nλ, Nφ, Hλ, Hφ;
                                    FT, latitude, longitude,
                                    Nproc, Nnum, a_curve, b_curve,
                                    first_pole_longitude)
    
    λFF = zeros(Nλ, Nφ)
    φFF = zeros(Nλ, Nφ)
    λFC = zeros(Nλ, Nφ)
    φFC = zeros(Nλ, Nφ)

    λCF = zeros(Nλ, Nφ)
    φCF = zeros(Nλ, Nφ)
    λCC = zeros(Nλ, Nφ)
    φCC = zeros(Nλ, Nφ)

    Lφ, φᵃᶠᵃ, φᵃᶜᵃ, Δφᵃᶠᵃ, Δφᵃᶜᵃ = generate_coordinate(FT, Periodic(), Nφ, Hφ, latitude,  :φ, CPU())
    Lλ, λᶠᵃᵃ, λᶜᵃᵃ, Δλᶠᵃᵃ, Δλᶜᵃᵃ = generate_coordinate(FT, Periodic(), Nλ, Hλ, longitude, :λ, CPU())
        
    # Identify equator line 
    J = Ref(0)
    for j in 1:Nφ+1
        if φᵃᶠᵃ[j] < 0
            J[] = j
        end
    end

    Jeq = J[] + 1

    aᶠⱼ = zeros(Nproc)
    bᶠⱼ = zeros(Nproc)
    cᶠⱼ = zeros(Nproc)

    aᶜⱼ = zeros(Nproc)
    bᶜⱼ = zeros(Nproc)
    cᶜⱼ = zeros(Nproc)

    xnum = zeros(Nλ+1, Nnum)
    ynum = zeros(Nλ+1, Nnum)
    jnum = zeros(Nλ+1, Nnum)

    fx = Float64.(collect(0:Nproc-1) ./ (Nproc-1) * Nφ .+ 1)

    φproc = range(φᵃᶠᵃ[1], 90 - Δφᵃᶠᵃ / 2, length = Nproc) 
    
    # calculate the eccentricities of the ellipse
    for (j, φ) in enumerate(φproc)
        aᶠⱼ[j] = a_curve(φ)
        bᶠⱼ[j] = b_curve(φ) 
    end
    
    a_face_interp(j) = linear_interpolate(j, fx, aᶠⱼ)
    b_face_interp(j) = linear_interpolate(j, fx, bᶠⱼ)

    φproc = range(φᵃᶜᵃ[1], 90, length = Nproc) 
    
    # calculate the eccentricities of the ellipse
    for (j, φ) in enumerate(φproc)
        aᶜⱼ[j] = a_curve(φ)
        bᶜⱼ[j] = b_curve(φ) 
    end

    a_center_interp(j) = linear_interpolate(j, fx, aᶜⱼ)
    b_center_interp(j) = linear_interpolate(j, fx, bᶜⱼ)

    # X - Face coordinates
    λ₀ = 90 # ᵒ degrees  

    # Face - Face coordinates
    loop! = compute_tripolar_coords!(device(CPU()), min(256, Nλ+1), Nλ+1)
    loop!(jnum, xnum, ynum, λ₀, Δλᶠᵃᵃ, 1/Nnum, Jeq, Nφ, a_face_interp, b_face_interp) 

    loop! = _compute_tripolar_coordinates!(device(CPU()), (16, 16), (Nλ, Nφ))
    loop!(λFF, φFF, Jeq, λ₀, Δλᶠᵃᵃ, φᵃᶠᵃ, a_curve, xnum, ynum, jnum, Nλ, Face())
    
    # Face - Center coordinates
    loop! = compute_tripolar_coords!(device(CPU()), min(256, Nλ+1), Nλ+1)
    loop!(jnum, xnum, ynum, λ₀, Δλᶠᵃᵃ, 1/Nnum, Jeq, Nφ, a_center_interp, b_center_interp) 

    loop! = _compute_tripolar_coordinates!(device(CPU()), (16, 16), (Nλ, Nφ))
    loop!(λFC, φFC, Jeq, λ₀, Δλᶠᵃᵃ, φᵃᶜᵃ, a_curve, xnum, ynum, jnum, Nλ, Center())
    
    # X - Center coordinates
    λ₀ = 90 + Δλᶜᵃᵃ / 2 # ᵒ degrees  

    # Center - Face  
    loop! = compute_tripolar_coords!(device(CPU()), min(256, Nλ+1), Nλ+1)
    loop!(jnum, xnum, ynum, λ₀, Δλᶜᵃᵃ, 1/Nnum, Jeq, Nφ, a_face_interp, b_face_interp) 

    loop! = _compute_tripolar_coordinates!(device(CPU()), (16, 16), (Nλ, Nφ))
    loop!(λCF, φCF, Jeq, λ₀, Δλᶜᵃᵃ, φᵃᶠᵃ, a_curve, xnum, ynum, jnum, Nλ, Face())
    
    # Face - Center coordinates
    loop! = compute_tripolar_coords!(device(CPU()), min(256, Nλ+1), Nλ+1)
    loop!(jnum, xnum, ynum, λ₀, Δλᶜᵃᵃ, 1/Nnum, Jeq, Nφ, a_center_interp, b_center_interp) 

    loop! = _compute_tripolar_coordinates!(device(CPU()), (16, 16), (Nλ, Nφ))
    loop!(λCC, φCC, Jeq, λ₀, Δλᶜᵃᵃ, φᵃᶜᵃ, a_curve, xnum, ynum, jnum, Nλ, Center())
    
    λmFC = deepcopy(λFC)
    λmCC = deepcopy(λCC) 

    for i in 1:Nλ÷2
        λFC[i, Nφ] = λmFC[Nλ - i + 1, Nφ]
        λCC[i, Nφ] = λmCC[Nλ - i + 1, Nφ]
    end

    for λ in (λFF, λFC, λCF, λCC)
        λ .+= first_pole_longitude 
        λ .=  convert_to_0_360.(λ)
    end

    return λFF, φFF, λFC, φFC, λCF, φCF, λCC, φCC
end

function insert_midpoint(λ, φ, λ₀, Δλ, Jeq, φ_grid, a_curve, Nλ, ::Center)
    λadd = zeros(Nλ)
    φadd = zeros(Nλ)

    for i in 1:Nλ
        j = Jeq + 1
        h = (λ₀ - Δλ * i) * 2π / 360
        x = - a_curve(φ_grid[j]) * cos(h)
        y = - a_curve(φ_grid[j]) * sin(h)

        λadd[i] =  - 180 / π * atan(y / x)
        φadd[i] =  90 - 360 / π * atan(sqrt(y^2 + x^2)) 

        λadd[i] += ifelse(i ≤ Nλ÷2, -90, 90)
    end

    λ = hcat(λ[:, 1:Jeq], λadd, λ[:, Jeq+1:end])
    φ = hcat(φ[:, 1:Jeq], φadd, φ[:, Jeq+1:end])

    return λ, φ
end

function insert_midpoint(λ, φ, λ₀, Δλ, Jeq, φ_grid, a_curve, Nλ, ::Face)
    λadd = zeros(Nλ)
    φadd = zeros(Nλ)

    for i in 1:Nλ
        j = Jeq + 1
        h = (λ₀ - Δλ * i) * 2π / 360
        x = - a_curve(φ_grid[j]) * cos(h)
        y = - a_curve(φ_grid[j]) * sin(h)

        λadd[i] =  - 180 / π * atan(y / x)
        φadd[i] =  90 - 360 / π * atan(sqrt(y^2 + x^2)) 

        λadd[i] += ifelse(i ≤ Nλ÷2, -90, 90)
    end

    λ = hcat(λ[:, 1:Jeq], λadd, λ[:, Jeq+1:end])
    φ = hcat(φ[:, 1:Jeq], φadd, φ[:, Jeq+1:end])

    return λ, φ
end

import Oceananigans.Grids: lat_lon_to_cartesian, lat_lon_to_x, lat_lon_to_y, lat_lon_to_z

function lat_lon_to_cartesian(lat, lon, radius)
    return [lat_lon_to_x(lat, lon, radius), lat_lon_to_y(lat, lon, radius), lat_lon_to_z(lat, lon, radius)]
end