
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

"""
    secant_root_find(j₀, j₁, f; tol = 1e-12)

Find the root of a function `f` using the secant method.

# Arguments
- `j₀`: Initial guess for the root.
- `j₁`: Second guess for the root.
- `f`: Function for which the root is to be found.
- `tol`: Tolerance for convergence. Default is `1e-12`.

# Returns
The approximate root of the function `f`.
"""
function secant_root_find(j₀, j₁, f, N; tol = 1e-12, maxiter=Inf)
    r = j₁ - f(j₁) * (j₁ - j₀) / (f(j₁) - f(j₀)) 
    iter = 0
    while abs(f(r)) > tol && iter < maxiter
        j₀ = max(1, min(j₁, N))
        j₁ = max(1, min(r,  N))
        r = j₁ - f(j₁) * (j₁ - j₀) / (f(j₁) - f(j₀)) 
        iter += 1
    end
    return r
end

function bisection_root_find(f, j₀, j₁, Δj)
    while j₀ + Δj < j₁ 
        jₘ = (j₀ + j₁) / 2
        if f(jₘ + 1) == 0
            return jₘ
        elseif f(jₘ + 1) < 0
            j₀ = jₘ
        else
            j₁ = jₘ
        end
    end
    return (j₀ + j₁) / 2
end

function get_wireframe(λF, φF)
    φF = 90 .- φF[1:end-1, :]
    λF = λF[1:end-1, :]

    x = @. cosd(λF) * sind(φF)
    y = @. sind(λF) * sind(φF)
    z = @. cosd(φF)

    return x, y, z
end

function haversine(a, b, radius)
    λ₁, φ₁ = a
    λ₂, φ₂ = b

    x₁, y₁, z₁ = lat_lon_to_cartesian(φ₁, λ₁, radius)
    x₂, y₂, z₂ = lat_lon_to_cartesian(φ₂, λ₂, radius)

    return radius * acos(max(-1.0, min((x₁ * x₂ + y₁ * y₂ + z₁ * z₂) / radius^2, 1.0)))
end

@kernel function _calculate_metrics!(Δxᶠᶜᵃ, Δxᶜᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                                     Δyᶠᶜᵃ, Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶠᵃ,
                                     Azᶠᶜᵃ, Azᶜᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ,
                                     λᶠᶜᵃ, λᶜᶜᵃ, λᶜᶠᵃ, λᶠᶠᵃ,
                                     φᶠᶜᵃ, φᶜᶜᵃ, φᶜᶠᵃ, φᶠᶠᵃ, radius)

    i, j = @index(Global, NTuple)
    @inbounds begin
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

@kernel function _compute_coordinates!(λF, φF, x, y, Jeq, Δλᶠᵃᵃ, φᵃᶠᵃ, f_curve, xnum, ynum, jnum)
    i, j = @index(Global, NTuple)

    @inbounds begin
        if j < Jeq
            h = (90 - Δλᶠᵃᵃ * i) * 2π / 360
            x[i, j] = - f_curve(φᵃᶠᵃ[j]) * cos(h)
            y[i, j] = - f_curve(φᵃᶠᵃ[j]) * sin(h)
        else
            x[i, j]  = linear_interpolate(j, jnum[i, :], xnum[i, :])
            y[i, j]  = linear_interpolate(j, jnum[i, :], ynum[i, :])
        end
        
        λF[i, j] = - 180 / π * (atan(y[i, j] / x[i, j]))              
        φF[i, j] = 90 - 360 / π * atan(sqrt(y[i, j]^2 + x[i, j]^2)) 
    end
end