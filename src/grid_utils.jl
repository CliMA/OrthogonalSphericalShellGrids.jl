
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
        @show r, j₀, j₁, abs(f(r))
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

function haversine(a, b, radius)
    λ₁, φ₁ = a
    λ₂, φ₂ = b

    x₁, y₁, z₁ = lat_lon_to_cartesian(φ₁, λ₁, radius)
    x₂, y₂, z₂ = lat_lon_to_cartesian(φ₂, λ₂, radius)

    return radius * acos(max(-1.0, min((x₁ * x₂ + y₁ * y₂ + z₁ * z₂) / radius^2, 1.0)))
end
