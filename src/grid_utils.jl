
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
function secant_root_find(j₀, j₁, f; tol = 1e-12)
    r = j₁ - f(j₁) * (j₁ - j₀) / (f(j₁) - f(j₀)) 
    while abs(f(r)) > tol
        j₀ = j₁
        j₁ = r
        r = j₁ - f(j₁) * (j₁ - j₀) / (f(j₁) - f(j₀)) 
    end
    return r
end

function haversine(a, b, radius)
    λ₁, φ₁ = a
    λ₂, φ₂ = b

    x₁, y₁, z₁ = lat_lon_to_cartesian(φ₁, λ₁, radius)
    x₂, y₂, z₂ = lat_lon_to_cartesian(φ₂, λ₂, radius)

    return radius * acos(max(-1.0, min((x₁ * x₂ + y₁ * y₂ + z₁ * z₂) / radius^2, 1.0)))
end
