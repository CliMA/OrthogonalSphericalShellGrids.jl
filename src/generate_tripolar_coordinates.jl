
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

        myfunc(x) = (xnum[i, 1] / a_interpolator(x)) ^2 + (ynum[i, 1] / b_interpolator(x))^2 - 1 
        jnum[i, 1] = bisection_root_find(myfunc, Jeq-1.0, Nφ+1-Δj, Δj)
        
        xnum[i, 2] = xnum[i, 1] - Δx
        ynum[i, 2] = ynum[i, 1] - Δx * tan(h)

        for n in 3:N
            # Great circles
            func(x) = (xnum[i, n-1] / a_interpolator(x)) ^2 + (ynum[i, n-1] / b_interpolator(x))^2 - 1 
            jnum[i, n-1] = bisection_root_find(func, Jeq-1.0, Nφ+1-Δj, Δj)
            xnum[i, n] = xnum[i, n-1] - Δx

            # Adams-Bashforth 2 integrator to find the perpendicular to the circle
            Gnew = ynum[i, n-1] * a_interpolator(jnum[i, n-1])^2 / b_interpolator(jnum[i, n-1])^2 / xnum[i, n-1]
            # Gold = ynum[i, n-2] * a_interpolator(jnum[i, n-2])^2 / b_interpolator(jnum[i, n-2])^2 / xnum[i, n-2]

            ynum[i, n] = ynum[i, n-1] - Δx * Gnew # - 0.6 * Gold)
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
function generate_tripolar_metrics!(λFF, φFF, λFC, φFC, λCF, φCF, λCC, φCC;
                                    FT, size, halo, latitude, longitude,
                                    Nproc, Nnum, a_curve, b_curve, c_curve,
                                    first_pole_longitude)

    Nλ, Nφ, Nz = size
    Hλ, Hφ, Hz = halo
                        
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

    aᶠⱼ = zeros(Nproc+1)
    bᶠⱼ = zeros(Nproc+1)
    cᶠⱼ = zeros(Nproc+1)

    aᶜⱼ = zeros(Nproc+1)
    bᶜⱼ = zeros(Nproc+1)
    cᶜⱼ = zeros(Nproc+1)

    xnum = zeros(Nλ+1, Nnum)
    ynum = zeros(Nλ+1, Nnum)
    jnum = zeros(Nλ+1, Nnum)

    φproc = range(φᵃᶠᵃ[1], φᵃᶠᵃ[Nφ+1], length = Nproc + 1) 
    
    # calculate the eccentricities of the ellipse
    for (j, φ) in enumerate(φproc)
        aᶠⱼ[j] = a_curve(φ)
        bᶠⱼ[j] = b_curve(φ) 
        cᶠⱼ[j] = c_curve(φ) 
    end

    fᶠx = Float64.(collect(0:Nproc) ./ Nproc * Nφ .+ 1)
    
    a_face_interp(j) = linear_interpolate(j, fᶠx, aᶠⱼ)
    b_face_interp(j) = linear_interpolate(j, fᶠx, bᶠⱼ)
    c_face_interp(j) = linear_interpolate(j, fᶠx, cᶠⱼ)

    φproc = range(φᵃᶜᵃ[1], φᵃᶜᵃ[Nφ+1], length = Nproc + 1) 
    
    # calculate the eccentricities of the ellipse
    for (j, φ) in enumerate(φproc)
        aᶜⱼ[j] = a_curve(φ)
        bᶜⱼ[j] = b_curve(φ) 
        cᶜⱼ[j] = c_curve(φ) 
    end

    a_center_interp(j) = linear_interpolate(j, fᶠx, aᶜⱼ)
    b_center_interp(j) = linear_interpolate(j, fᶠx, bᶜⱼ)
    c_center_interp(j) = linear_interpolate(j, fᶠx, cᶜⱼ)

    # X - Face coordinates
    λ₀ = 90 # ᵒ degrees  

    # Face - Face coordinates
    loop! = compute_tripolar_coords!(device(CPU()), min(256, Nλ+1), Nλ+1)
    loop!(jnum, xnum, ynum, λ₀, Δλᶠᵃᵃ, 1/Nnum, Jeq, Nφ, a_face_interp, b_face_interp, c_face_interp) 

    loop! = _compute_coordinates!(device(CPU()), (16, 16), (Nλ, Nφ))
    loop!(λFF, φFF, Jeq, λ₀, Δλᶠᵃᵃ, φᵃᶠᵃ, a_curve, xnum, ynum, jnum, Nλ)
    
    # Face - Center coordinates
    loop! = compute_tripolar_coords!(device(CPU()), min(256, Nλ+1), Nλ+1)
    loop!(jnum, xnum, ynum, λ₀, Δλᶠᵃᵃ, 1/Nnum, Jeq, Nφ, a_center_interp, b_center_interp, c_center_interp) 

    loop! = _compute_coordinates!(device(CPU()), (16, 16), (Nλ, Nφ))
    loop!(λFC, φFC, Jeq, λ₀, Δλᶠᵃᵃ, φᵃᶜᵃ, a_curve, xnum, ynum, jnum, Nλ)
    
    # X - Center coordinates
    λ₀ = 90 + Δλᶜᵃᵃ / 2 # ᵒ degrees  

    # Center - Face 
    loop! = compute_tripolar_coords!(device(CPU()), min(256, Nλ+1), Nλ+1)
    loop!(jnum, xnum, ynum, λ₀, Δλᶜᵃᵃ, 1/Nnum, Jeq, Nφ, a_face_interp, b_face_interp, c_face_interp) 
    
    loop! = _compute_coordinates!(device(CPU()), (16, 16), (Nλ, Nφ))
    loop!(λCF, φCF, Jeq, λ₀, Δλᶜᵃᵃ, φᵃᶠᵃ, a_curve, xnum, ynum, jnum, Nλ)
    
    # Face - Center coordinates
    loop! = compute_tripolar_coords!(device(CPU()), min(256, Nλ+1), Nλ+1)
    loop!(jnum, xnum, ynum, λ₀, Δλᶜᵃᵃ, 1/Nnum, Jeq, Nφ, a_center_interp, b_center_interp, c_center_interp) 

    loop! = _compute_coordinates!(device(CPU()), (16, 16), (Nλ, Nφ))
    loop!(λCC, φCC, Jeq, λ₀, Δλᶜᵃᵃ, φᵃᶜᵃ, a_curve, xnum, ynum, jnum, Nλ)

    # Metrics
    for λ in (λFF, λFC, λCF, λCC)
        # λ .+= first_pole_longitude 
        λ .=  convert_to_0_360.(λ)
    end

    return nothing
end