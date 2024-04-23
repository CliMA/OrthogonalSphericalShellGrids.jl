
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

const γ¹ = 8 // 15
const γ² = 5 // 12
const γ³ = 3 // 4

const ζ² = -17 // 60
const ζ³ = -5 // 12

@kernel function _compute_tripolar_coords!(λ, φ, λᵢ, φᵢ, Δλ, Δφ, Jeq, Nφ, Nλ, Nnum, a_curve, b_curve, a_integrator, b_integrator)
    i = @index(Global, Linear)
    h = (λᵢ - Δλ * i) * 2π / 360
    @inbounds begin
        for j = 1:Jeq
            φ[i, j] = φᵢ + Δφ * (j - 1)
            x = - a_curve(φ[i, Jeq]) * cos(h)
            y = - a_curve(φ[i, Jeq]) * sin(h)
            λ[i, j] = - 180 / π * ifelse(x == 0, ifelse(y == 0, 0, - atan(Base.sign(y) * Inf)), atan(y / x))
            λ[i, j] += ifelse(i ≤ Nλ÷2, -90, 90)
        end

        # Starting from Jeq + 1
        x = - a_curve(φ[i, Jeq]) * cos(h)
        y = - a_curve(φ[i, Jeq]) * sin(h)

        Neq = Nφ - Jeq + 1

        # Number of integration steps per step
        Δx = x / (Neq * Nnum)

        for j in Jeq:Nφ
            for n in 1:Nnum
                # Great circles
                func(j) = (x / a_integrator(j)) ^2 + (y / b_integrator(j))^2 - 1 
                jn = bisection_root_find(func, Jeq-1.0, Nφ+1.0, 1/Nnum)
    
                # Semi-Implicit integrator
                # φₙ = 90 - 360 / π * atan(sqrt(y^2 + x^2)) 
                G¹ = a_integrator(jn)^2 / b_integrator(jn)^2 
                
                y  = y / (1 + G¹ / x * Δx) 
                x -= Δx

                # Runge Kutta integrator to find the perpendicular to the circle

                # # Fist substep!
                # G² = y * a_curve(φₙ)^2 / b_curve(φₙ)^2 / x
                # x -= Δx * γ¹ 
                # y -= Δx * γ¹ * G²
                # φₙ = 90 - 360 / π * atan(sqrt(y^2 + x^2)) 

                # # Second substep!
                # G¹ = G²
                # G² = y * a_curve(φₙ)^2 / b_curve(φₙ)^2/ x
                # x -= Δx * (γ² + ζ²)  
                # y -= Δx * (γ² * G² + ζ² * G¹)   
                # φₙ = 90 - 360 / π * atan(sqrt(y^2 + x^2)) 

                # # Third substep!
                # G¹ = G²
                # G² = y * a_curve(φₙ)^2 / b_curve(φₙ)^2 / x
                # x -= Δx * (γ³ + ζ³)  
                # y -= Δx * (γ³ * G² + ζ³ * G¹)  
                
                if i == Nλ
                    @show x, y, G¹
                end  
            end

            λ[i, j] = - 180 / π * ifelse(x == 0, ifelse(y == 0, 0, - atan(Base.sign(y) * Inf)), atan(y / x))
            φ[i, j] = 90 - 360 / π * atan(sqrt(y^2 + x^2)) 
            λ[i, j] += ifelse(i ≤ Nλ÷2, -90, 90)
            if i == Nλ || i == Nλ-1
                @show λ[i, j], φ[i, j], i
            end
        end

        # @show i, x
    end
end

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

    fx = Float64.(collect(0:Nproc-1) ./ (Nproc-1) * Nφ .+ 1)

    φFproc = range(φᵃᶠᵃ[1], 90 - Δφᵃᶠᵃ / 2, length = Nproc) 
    Δφᵃᶠᵃ = (φFproc[2] - φFproc[1]) .* Nproc / Nφ
     
    # calculate the eccentricities of the ellipse
    for (j, φ) in enumerate(φproc)
        aᶠⱼ[j] = a_curve(φ)
        bᶠⱼ[j] = b_curve(φ) 
    end
    
    a_face_interp(j) = linear_interpolate(j, fx, aᶠⱼ)
    b_face_interp(j) = linear_interpolate(j, fx, bᶠⱼ)

    φCproc = range(φᵃᶜᵃ[1], 90, length = Nproc) 
    Δφᵃᶜᵃ = (φCproc[2] - φCproc[1]) .* Nproc / Nφ

    # calculate the eccentricities of the ellipse
    for (j, φ) in enumerate(φproc)
        aᶜⱼ[j] = a_curve(φ)
        bᶜⱼ[j] = b_curve(φ) 
    end

    a_center_interp(j) = linear_interpolate(j, fx, aᶜⱼ)
    b_center_interp(j) = linear_interpolate(j, fx, bᶜⱼ)

    integration_kernel! = _compute_tripolar_coords!(device(CPU()), min(256, Nλ+1), Nλ+1)
    
    # # X - Face coordinates
    λ₀ = 90 # ᵒ degrees  

    # # Face - Face coordinates    
    φ₀ = φFproc[1]
    integration_kernel!(λFF, φFF, λ₀, φ₀, Δλᶠᵃᵃ, Δφᵃᶠᵃ, Jeq, Nφ, Nλ, 100, a_curve, b_curve, a_face_interp, b_face_interp) 

    # # Face - Center coordinates
    φ₀ = φCproc[1]
    integration_kernel!(λFC, φFC, λ₀, φ₀, Δλᶠᵃᵃ, Δφᵃᶜᵃ, Jeq, Nφ, Nλ, 100, a_curve, b_curve, a_face_interp, b_face_interp) 

    # X - Center coordinates
    λ₀ = 90 + Δλᶜᵃᵃ / 2 # ᵒ degrees  

    # Center - Face  
    φ₀ = φFproc[1]
    integration_kernel!(λCF, φCF, λ₀, φ₀, Δλᶜᵃᵃ, Δφᵃᶠᵃ, Jeq, Nφ, Nλ, 100, a_curve, b_curve, a_center_interp, b_center_interp)
    
    # Face - Center coordinates
    φ₀ = φCproc[1]
    integration_kernel!(λCC, φCC, λ₀, φ₀, Δλᶜᵃᵃ, Δφᵃᶜᵃ, Jeq, Nφ, Nλ, 100, a_curve, b_curve, a_center_interp, b_center_interp)

    for λ in (λFF, λFC, λCF, λCC)
        λ .+= first_pole_longitude 
        λ .=  convert_to_0_360.(λ)
    end

    return λFF, φFF, λFC, φFC, λCF, φCF, λCC, φCC
end

# function calculate_face_coords!(λFF, φFF, λFC, φFC, λCF, φCF, λCC, φCC)
#     x, y = stereographic_projection.(λCF, φCF)



# import Oceananigans.Grids: lat_lon_to_cartesian, lat_lon_to_x, lat_lon_to_y, lat_lon_to_z

# function stereographic_projection(λ, φ)

# end

function lat_lon_to_cartesian(lat, lon, radius)
    return [lat_lon_to_x(lat, lon, radius), lat_lon_to_y(lat, lon, radius), lat_lon_to_z(lat, lon, radius)]
end