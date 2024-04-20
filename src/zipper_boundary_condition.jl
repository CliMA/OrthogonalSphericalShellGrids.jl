using Oceananigans.Grids: Center, Face
using Oceananigans.Utils: KernelParameters, launch!

import Oceananigans.BoundaryConditions: bc_str, _fill_north_halo!, apply_y_north_bc!
using Oceananigans.BoundaryConditions: AbstractBoundaryConditionClassification, BoundaryCondition
import Oceananigans.Fields: validate_boundary_condition_location

# currently only supporting MITgcm "jperio=4" north folds (ORCA 2, 1/4 and 1/12)
struct Zipper <: AbstractBoundaryConditionClassification end

ZipperBoundaryCondition(sign = 1) = BoundaryCondition(Zipper, sign)

bc_str(zip::Zipper) = "Zipping boundary"

const ZBC = BoundaryCondition{<:Zipper}

apply_y_north_bc!(Gc, loc, ::ZBC, args...) = nothing

validate_boundary_condition_location(bc::Zipper, loc::Center, side) = 
    side == :north ? nothing : throw(ArgumentError("Cannot specify $side boundary condition $bc on a field at $(loc) (north only)!"))

validate_boundary_condition_location(bc::Zipper, loc::Face, side) = 
    side == :north ? nothing : throw(ArgumentError("Cannot specify $side boundary condition $bc on a field at $(loc) (north only)!"))

@inline apply_y_north_bc!(Gc, loc, ::Zipper, args...) = nothing

#####
##### Outer functions for filling halo regions for Zipper boundary conditions.
#####

@inline function fold_north_face!(i, k, grid, sign, c)
    Nx, Ny, _ = size(grid)
    
    i′ = Nx - i + 2
    Hy = grid.Hy
    
    for j = 1 : Hy
        @inbounds begin
            c[i, Ny + j, k] = sign * c[i′, Ny - j + 1, k] 
        end
    end

    return nothing
end

@inline function fold_north_center!(i, k, grid, sign, c)
    Nx, Ny, _ = size(grid)
    
    i′ = Nx - i + 1
    Hy = grid.Hy
    
    for j = 1 : Hy
        @inbounds begin
            c[i, Ny + j, k] = sign * c[i′, Ny - j + 1, k] 
        end
    end

    return nothing
end

const CLocation = Union{Tuple{<:Center}, Tuple{<:Center, <:Any}, Tuple{<:Center, <:Any, <:Any}} 
const FLocation = Union{Tuple{<:Face},   Tuple{<:Face, <:Any},   Tuple{<:Face, <:Any, <:Any}}

# u-velocity or similar fields
@inline _fill_north_halo!(i, k, grid, c, bc::ZBC, ::CLocation, args...) = fold_north_center!(i, k, grid, bc.condition, c)

# v-velocity or similar fields
@inline _fill_north_halo!(i, k, grid, c, bc::ZBC, ::FLocation, args...) = fold_north_face!(i, k, grid, bc.condition, c)
