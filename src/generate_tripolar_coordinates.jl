"""
    _compute_tripolar_coordinates!(λFF, φFF, λFC, φFC, λCF, φCF, λCC, φCC, 
                                   λᶠᵃᵃ, λᶜᵃᵃ, φᵃᶠᵃ, φᵃᶜᵃ, 
                                   first_pole_longitude,
                                   focal_distance, Nλ)

Compute the tripolar coordinates for a given set of input parameters. This function follows the formulation described in 
`Ross J. Murray, "Explicit Generation of Orthogonal Grids for Ocean Models", Journal of Computational Physics,
Volume 126, Issue 2, 1996, Pages 251-273`.

The tripolar grid is built as a set of cofocal ellipsed and perpendicular hyperbolae.
The `focal_distance` argument is the distance from the center of the ellipses to the foci.

The ellipses obeys:

       x²          y²
   --------- + ---------  = 1
   a²cosh²(ψ)  a²sinh²(ψ)

While the set of perpendicular hyperbolae obey:

       x²          y²
   --------- + ---------  = 1
   a²cos²(λ)   a²sin²(λ)

Where `a` is the `focal_distance` to the center. `λ` is the longitudinal angle and `ψ` is the ``isometric latitude'' 
defined by Murray (1996) as satisfying:

    a sinh(ψ) = tand((90 - φ) / 2) 

The final (x, y) points that define the stereographic projection of the tripolar coordinates are given by:

    x = a * sinh(ψ) * cos(λ)
    y = a * sinh(ψ) * sin(λ)

for which it is possible to retrive the longitude and latitude by:

    λ = - 180 / π * atan(y / x)
    φ = 90 - 360 / π * atan(sqrt(y² + x²))
"""
@kernel function _compute_tripolar_coordinates!(λFF, φFF, λFC, φFC, λCF, φCF, λCC, φCC, 
                                                λᶠᵃᵃ, λᶜᵃᵃ, φᵃᶠᵃ, φᵃᶜᵃ, 
                                                first_pole_longitude,
                                                focal_distance, Nλ)
    
    i, j = @index(Global, NTuple)

    λ2Ds = (λFF,  λFC,  λCF,  λCC)
    φ2Ds = (φFF,  φFC,  φCF,  φCC)
    λ1Ds = (λᶠᵃᵃ, λᶠᵃᵃ, λᶜᵃᵃ, λᶜᵃᵃ)
    φ1Ds = (φᵃᶜᵃ, φᵃᶠᵃ, φᵃᶜᵃ, φᵃᶠᵃ)

    for (λ2D, φ2D, λ1D, φ1D) in zip(λ2Ds, φ2Ds, λ1Ds, φ1Ds)
        ψ = asinh(tand((90 - φ1D[j]) / 2) / focal_distance)
        x = focal_distance * sind(λ1D[i]) * cosh(ψ)
        y = focal_distance * cosd(λ1D[i]) * sinh(ψ)

        # When x == 0 and y == 0 we are exactly at the north pole,
        # and λ (which depends on `atan(y / x)`) is not defined
        # This makes sense, what is the longitude of the north pole? Could be anything!
        # so we choose a value that is continuos with the surrounding points.
        on_the_north_pole = (x == 0) & (y == 0)
        north_pole_value  = ifelse(i == 1, -90, 90) 

        λ2D[i, j] = ifelse(on_the_north_pole, north_pole_value, - 180 / π * atan(y / x))
        φ2D[i, j] = 90 - 360 / π * atan(sqrt(y^2 + x^2)) # The latitude will be in the range [-90, 90]

        # Shift longitude to the range [-180, 180], the 
        # the north singluarities are located at -180 and 0
        λ2D[i, j] += ifelse(i ≤ Nλ÷2, -90, 90) 

        # Make sure out singularities are aligned with the 
        # longitude we want them to be at. The singularities are now are at -180 and 0,
        # we want them to be at first_pole_longitude and first_pole_longitude + 180
        λ2D[i, j] += first_pole_longitude 
        λ2D[i, j]  = convert_to_0_360(λ2D[i, j])
    end
end