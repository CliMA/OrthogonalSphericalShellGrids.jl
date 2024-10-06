using OrthogonalSphericalShellGrids
using OrthogonalSphericalShellGrids: TRG
using Oceananigans
using Oceananigans.Operators: Δx, Δy
using Oceananigans.Grids: λnode, φnode, znode
using Oceananigans.Fields: fractional_index, fractional_z_index, AbstractField, interpolator

import Oceananigans.Fields: interpolate!, λnodes, φnodes

# TODO: Move to Oceananigans
@inline λnodes(ibg::ImmersedBoundaryGrid, args...; kwargs...) = λnodes(ibg.underlying_grid, args...; kwargs...)
@inline φnodes(ibg::ImmersedBoundaryGrid, args...; kwargs...) = φnodes(ibg.underlying_grid, args...; kwargs...)

TRGField = Field{<:Any, <:Any, <:Any, <:Any, <:TRG}

#####
##### Nearest Neighbor Interpolation from a Tripolar Field to a Latitude Longitude Field
#####

# This is a "naive" algorithm, intended only for visualization and zonal averages, not for more sofisticated diagnostics. 
# Optimizations and improvements are welcome.
# Maybe move this to Oceananigans? Probably it requires a bit more validation...

struct InterpolationWeights{LXT, LYT, LXF, LYF, I, J, W}
    i_indices :: I
    j_indices :: J
    weights   :: W

    function InterpolationWeights{LXT, LYT, LXF, LYF}(i::I, j::J, w::W) where {LXT, LYT, LXF, LYF, I, J, W} 
        return new{LXT, LYT, LXF, LYF, I, J, W}(i, j, w)
    end
end

@inline from_location(::InterpolationWeights{LXT, LYT, LXF, LYF}) where {LXT, LYT, LXF, LYF} = (LXF, LYF)
@inline to_location(::InterpolationWeights{LXT, LYT, LXF, LYF}) where {LXT, LYT, LXF, LYF} = (LXT, LYT)

Adapt.adapt_structure(to, iw::InterpolationWeights{LXT, LYT, LXF, LYF}) where {LXT, LYT, LXF, LYF} = 
    InterpolationWeights{LXT, LYT, LXF, LYF}(Adapt.adapt(to, iw.i_indices),
                                             Adapt.adapt(to, iw.j_indices),
                                             Adapt.adapt(to, iw.weights))

function InterpolationWeights(to_field::Field, from_field::TRGField)

    to_grid = to_field.grid
    from_grid = from_field.grid
    
    Nx, Ny, _ = size(to_field)
    arch = architecture(to_grid)

    i_indices = on_architecture(arch, zeros(Int, Nx, Ny))
    j_indices = on_architecture(arch, zeros(Int, Nx, Ny))
    weights   = on_architecture(arch, zeros(eltype(to_grid), Nx, Ny, 9))

    to_loc   = location(to_field)
    from_loc = location(from_field)

    launch!(arch, to_grid, (Nx, Ny), _compute_weights!, 
            i_indices, j_indices, weights, 
            to_grid, from_grid, map(instantiate, to_loc), map(instantiate, from_loc))

    return InterpolationWeights{to_loc[1], to_loc[2], from_loc[1], from_loc[2]}(i_indices, j_indices, weights)
end

@kernel function _compute_weights!(i_indices, j_indices, weights, to_grid, from_grid, to_loc, from_loc)
    i, j = @index(Global, NTuple)

    λ₀ = λnode(i, j, 1, to_grid, to_loc...)
    φ₀ = φnode(i, j, 1, to_grid, to_loc...)
    i₀, j₀, d₀₀, d₀₁, d₁₀, d₀₂, d₂₀, d₁₁, d₂₂, d₁₂, d₂₁ = horizontal_distances(λ₀, φ₀, from_loc, from_grid)

    @inbounds begin
        i_indices[i, j] = i₀
        j_indices[i, j] = j₀

        weights[i, j, 1] = 1 / d₀₀
        weights[i, j, 2] = 1 / d₀₁ 
        weights[i, j, 3] = 1 / d₁₀ 
        weights[i, j, 4] = 1 / d₀₂ 
        weights[i, j, 5] = 1 / d₂₀ 
        weights[i, j, 6] = 1 / d₁₁ 
        weights[i, j, 7] = 1 / d₂₂ 
        weights[i, j, 8] = 1 / d₁₂ 
        weights[i, j, 9] = 1 / d₂₁
    end
end

function interpolate!(to_field::Field, from_field::TRGField, interpolation_weights = nothing)

    to_loc = location(to_field)
    from_loc = location(from_field)

    # Make sure weights are correct
    # TODO: a check also on the grid (to and from)
    if !(interpolation_weights isa InterpolationWeights)
        interpolation_weights = InterpolationWeights(to_field, from_field)
    else
        # Check that the locations are correct
        LXF, LYF = from_location(interpolation_weights)
        LXT, LYT = to_location(interpolation_weights)

        correct_locations = (LXF, LYF) == from_loc[1:2] && (LXT, LYT) == to_loc[1:2]
        
        if !correct_locations 
            throw("The location of the interpolation weights do not coincide with the locations of the in and out fields")
        end
    end

    to_loc   = map(instantiate, to_loc)
    from_loc = map(instantiate, from_loc)

    to_grid   = to_field.grid
    from_grid = from_field.grid

    to_arch   = architecture(to_field)
    from_arch = architecture(from_field)

    # Make sure `from_field` has the boundary conditions filled
    fill_halo_regions!(from_field)

    # In case architectures are `Distributed` we
    # verify that the fields are on the same child architecture
    to_arch   = child_architecture(to_arch)
    from_arch = child_architecture(from_arch)

    if !isnothing(from_arch) && to_arch != from_arch
        msg = "Cannot interpolate! because from_field is on $from_arch while to_field is on $to_arch."
        throw(ArgumentError(msg))
    end

    launch!(to_arch, to_grid, size(to_field),
            _nearest_neigbor_interpolate!, to_field, to_loc, to_grid, from_field, from_loc, from_grid, interpolation_weights)

    fill_halo_regions!(to_field)

    return to_field
end

@kernel function _nearest_neigbor_interpolate!(to_field, to_loc, to_grid, from_field, from_loc, from_grid, interpolation_weights)
    i, j, k = @index(Global, NTuple)

    z  = znode(k, to_grid, to_loc[3])
    kk = fractional_z_index(z, from_loc, from_grid)
    
    k⁻, k⁺, ζ = interpolator(kk)

    ϕ⁻ = horizontal_interpolate(i, j, k⁻, from_grid, from_field, interpolation_weights)
    ϕ⁺ = horizontal_interpolate(i, j, k⁺, from_grid, from_field, interpolation_weights)

    @inbounds to_field[i, j, k] = ϕ⁻ * (1 - ζ) + ϕ⁺ * ζ
end

@inline function horizontal_interpolate(i, j, k, grid, from_field, iw)

    i₀ = @inbounds iw.i_indices[i, j]
    j₀ = @inbounds iw.j_indices[i, j]

    i₁ = ifelse(i₀ == 0, size(from_field, 1),  i₀ - 1)
    j₁ = ifelse(j₀ == 0, j₀, j₀ - 1)
    i₂ = ifelse(i₀ == size(from_field, 1), 1,  i₀ + 1)
    j₂ = ifelse(j₀ == size(from_field, 2), j₀, j₀ + 1)

    @inbounds begin
        f₀₀ = from_field[i₀, j₀, k]
        f₀₁ = from_field[i₀, j₁, k]
        f₁₀ = from_field[i₁, j₀, k]
        f₀₂ = from_field[i₀, j₂, k]
        f₂₀ = from_field[i₂, j₀, k]
        f₁₁ = from_field[i₁, j₁, k]
        f₂₂ = from_field[i₂, j₂, k]
        f₁₂ = from_field[i₁, j₂, k]
        f₂₁ = from_field[i₂, j₁, k]

        w₀₀ = iw.weights[i, j, 1]
        w₀₁ = iw.weights[i, j, 2]
        w₁₀ = iw.weights[i, j, 3]
        w₀₂ = iw.weights[i, j, 4]
        w₂₀ = iw.weights[i, j, 5]
        w₁₁ = iw.weights[i, j, 6]
        w₂₂ = iw.weights[i, j, 7]
        w₁₂ = iw.weights[i, j, 8]
        w₂₁ = iw.weights[i, j, 9]
    end

    ∑fw = f₀₀ * w₀₀ + f₀₁ * w₀₁ + f₁₀ * w₁₀ + f₀₂ * w₀₂ + f₂₀ * w₂₀ + f₁₁ * w₁₁ + f₂₂ * w₂₂ + f₁₂ * w₁₂ + f₂₁ * w₂₁
    ∑w = w₀₀ + w₀₁ + w₁₀ + w₀₂ + w₂₀ + w₁₁ + w₂₂ + w₁₂ + w₂₁

    return ∑fw / ∑w
end

#####
##### Weight computation
#####

@inline function distance(x₁, y₁, x₂, y₂) 
    dx = x₁ - x₂
    dy = y₁ - y₂
    return dx * dx + dy * dy
end

@inline function check_and_update(dist, i₀, j₀, i, j, λ₀, φ₀, λ, φ)               
    d = distance(λ₀, φ₀, λ , φ) 
    i₀ = ifelse(d < dist, i, i₀)          
    j₀ = ifelse(d < dist, j, j₀)          
    dist = min(d, dist)

    return dist, i₀, j₀
end

# # We assume that in an TRG, the latitude lines for a given i - index are sorted
# # i.e. φ is monotone in j. This is not the case for λ that might jump between 0 and 360.
@inline function horizontal_distances(λ₀, φ₀, loc, grid)

    λ = λnodes(grid, loc...; with_halos = true)
    φ = φnodes(grid, loc...; with_halos = true)

    Nx, Ny, _ = size(grid)

    # We search for an initial valid option
    dist = Inf
    i₀ = 1
    j₀ = 1

    @inbounds begin
        for i = 1:Nx
            φi = view(φ, i, :)
            jⁿ = fractional_index(φ₀, φi, Ny) - 1
            j⁻ = floor(Int, jⁿ)
            j⁺ = j⁻ + 1

            if j⁻ <= grid.Ny
                dist, i₀, j₀ = check_and_update(dist, i₀, j₀, i, j⁻, λ₀, φ₀, λ[i, j⁻], φ[i, j⁻])               
            end

            if j⁺ <= grid.Ny
                dist, i₀, j₀ = check_and_update(dist, i₀, j₀, i, j⁺, λ₀, φ₀, λ[i, j⁺], φ[i, j⁺])
            end
        end 
    end
    
    # Now find the closest neighbors given i₀ and j₀
    i₁ = ifelse(i₀ == 0, grid.Nx, i₀ - 1)
    j₁ = ifelse(j₀ == 0, j₀, j₀ - 1)
    i₂ = ifelse(i₀ == size(λ, 1), 1,  i₀ + 1)
    j₂ = ifelse(j₀ == size(λ, 2), j₀, j₀ + 1)

    @inbounds begin
        λ₀₀ = massage_longitude(λ₀, λ[i₀, j₀])
        λ₀₁ = massage_longitude(λ₀, λ[i₀, j₁])
        λ₁₀ = massage_longitude(λ₀, λ[i₁, j₀])
        λ₀₂ = massage_longitude(λ₀, λ[i₀, j₂])
        λ₂₀ = massage_longitude(λ₀, λ[i₂, j₀])
        λ₁₁ = massage_longitude(λ₀, λ[i₁, j₁])
        λ₂₂ = massage_longitude(λ₀, λ[i₂, j₂])
        λ₁₂ = massage_longitude(λ₀, λ[i₁, j₂])
        λ₂₁ = massage_longitude(λ₀, λ[i₂, j₁])
        
        φ₀₀ = φ[i₀, j₀]
        φ₀₁ = φ[i₀, j₁]
        φ₁₀ = φ[i₁, j₀]
        φ₀₂ = φ[i₀, j₂]
        φ₂₀ = φ[i₂, j₀]
        φ₁₁ = φ[i₁, j₁]
        φ₂₂ = φ[i₂, j₂]
        φ₁₂ = φ[i₁, j₂]
        φ₂₁ = φ[i₂, j₁]
    end

    d₀₀ = distance(λ₀, φ₀, λ₀₀, φ₀₀)
    d₀₁ = distance(λ₀, φ₀, λ₀₁, φ₀₁)
    d₁₀ = distance(λ₀, φ₀, λ₁₀, φ₁₀)
    d₀₂ = distance(λ₀, φ₀, λ₀₂, φ₀₂)
    d₂₀ = distance(λ₀, φ₀, λ₂₀, φ₂₀)
    
    d₁₁ = distance(λ₀, φ₀, λ₁₁, φ₁₁)
    d₂₂ = distance(λ₀, φ₀, λ₂₂, φ₂₂)
    d₁₂ = distance(λ₀, φ₀, λ₁₂, φ₁₂)
    d₂₁ = distance(λ₀, φ₀, λ₂₁, φ₂₁)

    return i₀, j₀, d₀₀, d₀₁, d₁₀, d₀₂, d₂₀, d₁₁, d₂₂, d₁₂, d₂₁
end

# We assume that all points are very close to each other, so a longitude difference of 180 should not possible, 
# this means that the we are on the same side of the globe, but that the longitude is displaced by 360 degrees.
@inline massage_longitude(λ₀, λ) = ifelse(abs(λ₀ - λ) > 180, 
                                   ifelse(λ₀ > 180, λ + 360, λ - 360), λ)