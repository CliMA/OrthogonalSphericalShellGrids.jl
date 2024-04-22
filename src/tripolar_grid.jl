"""
    struct WarpedLatitudeLongitude

A struct representing a warped latitude-longitude grid.

TODO: put here information about the grid, i.e.: 

1) north pole latitude and longitude
2) functions used to construct the Grid
3) Numerical discretization used to construct the Grid
4) Last great circle size in degrees
"""
struct Tripolar end

@inline tripolar_stretching_function(φ; d = 0.4) = (φ / 90)^4 * d

@inline cosine_a_curve(φ)          = - equator_fcurve(φ) 
@inline cosine_b_curve(φ; d = 0.4) = - equator_fcurve(φ) + ifelse(φ > 0, tripolar_stretching_function(φ; d), 0)

@inline zero_c_curve(φ) = 0

const TripolarGrid{FT, TX, TY, TZ, A, R, FR, Arch} = OrthogonalSphericalShellGrid{FT, TX, TY, TZ, A, R, FR, <:Tripolar, Arch}

"""
    TripolarGrid(arch = CPU(), FT::DataType = Float64; 
                 size, 
                 southermost_latitude = -75, 
                 halo        = (4, 4, 4), 
                 radius      = R_Earth, 
                 z           = (0, 1),
                 singularity_longitude = 230,
                 f_curve     = quadratic_f_curve,
                 g_curve     = quadratic_g_curve)

Constructs a tripolar grid on a spherical shell.

Positional Arguments
====================

- `arch`: The architecture to use for the grid. Default is `CPU()`.
- `FT::DataType`: The data type to use for the grid. Default is `Float64`.

Keyword Arguments
=================

- `size`: The number of cells in the (longitude, latitude, z) dimensions.
- `southermost_latitude`: The southernmost latitude of the grid. Default is -75.
- `halo`: The halo size in the (longitude, latitude, z) dimensions. Default is (4, 4, 4).
- `radius`: The radius of the spherical shell. Default is `R_Earth`.
- `z`: The z-coordinate range of the grid. Default is (0, 1).
- `singularity_longitude`: The longitude at which the grid has a singularity. Default is 230.
- `f_curve`: The function to compute the f-curve for the grid. Default is `quadratic_f_curve`.
- `g_curve`: The function to compute the g-curve for the grid. Default is `quadratic_g_curve`.

Returns
========

A `OrthogonalSphericalShellGrid` object representing a tripolar grid on the sphere
"""
function TripolarGrid(arch = CPU(), FT::DataType = Float64; 
                      size, 
                      southermost_latitude = -82, 
                      halo                 = (4, 4, 4), 
                      radius               = R_Earth, 
                      z                    = (0, 1),
                      poles_latitude       = 45,
                      first_pole_longitude = 75,    # The second pole will be at `λ = first_pole_longitude + 180ᵒ`
                      Nproc                = 1000, 
                      Nnum                 = 1000, 
                      a_curve              = cosine_a_curve,
                      initial_b_curve      = cosine_b_curve,
                      c_curve              = zero_c_curve)

    # For now, only for domains Periodic in λ (from -180 to 180 degrees) and Bounded in φ.
    # φ has to reach the north pole.`
    # For all the rest we can easily use a `LatitudeLongitudeGrid` without warping
    final_b    = sqrt((tan((90 - poles_latitude) / 360 * π))^2)
    b_curve(φ) = initial_b_curve(φ; d = final_b)

    latitude  = (southermost_latitude, 90)
    longitude = (-180, 180) 
    
    Nλ, Nφ, Nz = size
    Hλ, Hφ, Hz = halo

    # the Z coordinate is the same as for the other grids
    Lz, zᵃᵃᶠ, zᵃᵃᶜ, Δzᵃᵃᶠ, Δzᵃᵃᶜ = generate_coordinate(FT, Bounded(),  Nz, Hz, z, :z, CPU())

    λFF = zeros(Nλ, Nφ+1)
    φFF = zeros(Nλ, Nφ+1)
    λFC = zeros(Nλ, Nφ+1)
    φFC = zeros(Nλ, Nφ+1)

    λCF = zeros(Nλ, Nφ+1)
    φCF = zeros(Nλ, Nφ+1)
    λCC = zeros(Nλ, Nφ+1)
    φCC = zeros(Nλ, Nφ+1)

    generate_tripolar_metrics!(λFF, φFF, λFC, φFC, λCF, φCF, λCC, φCC;
                               FT, size, halo, latitude, longitude,
                               Nproc, Nnum, a_curve, b_curve, c_curve,
                               first_pole_longitude)

    Nx = Nλ
    Ny = Nφ
    
    # Helper grid to fill halo 
    grid = RectilinearGrid(; size = (Nx, Ny, 1), halo, topology = (Periodic, RightConnected, Bounded), z = (0, 1), x = (0, 1), y = (0, 1))

    default_boundary_conditions = FieldBoundaryConditions(north  = ZipperBoundaryCondition(),
                                                          south  = nothing, # The south should be `continued`
                                                          west   = Oceananigans.PeriodicBoundaryCondition(),
                                                          east   = Oceananigans.PeriodicBoundaryCondition(),
                                                          top    = nothing,
                                                          bottom = nothing)
                                                        
    lFF = Field((Face, Face, Center), grid; boundary_conditions = default_boundary_conditions)
    pFF = Field((Face, Face, Center), grid; boundary_conditions = default_boundary_conditions)

    lFC = Field((Face, Center, Center), grid; boundary_conditions = default_boundary_conditions)
    pFC = Field((Face, Center, Center), grid; boundary_conditions = default_boundary_conditions)
    
    lCF = Field((Center, Face, Center), grid; boundary_conditions = default_boundary_conditions)
    pCF = Field((Center, Face, Center), grid; boundary_conditions = default_boundary_conditions)

    lCC = Field((Center, Center, Center), grid; boundary_conditions = default_boundary_conditions)
    pCC = Field((Center, Center, Center), grid; boundary_conditions = default_boundary_conditions)

    set!(lFF, λFF[:, 1:Ny])
    set!(pFF, φFF[:, 1:Ny])

    set!(lFC, λFC[:, 1:Ny])
    set!(pFC, φFC[:, 1:Ny])

    set!(lCF, λCF[:, 1:Ny])
    set!(pCF, φCF[:, 1:Ny])

    set!(lCC, λCC[:, 1:Ny])
    set!(pCC, φCC[:, 1:Ny])

    fill_halo_regions!(lFF)
    fill_halo_regions!(lCF)
    fill_halo_regions!(lFC)
    fill_halo_regions!(lCC)
    
    fill_halo_regions!(pFF)
    fill_halo_regions!(pCF)
    fill_halo_regions!(pFC)
    fill_halo_regions!(pCC)

    λᶠᶠᵃ = lFF.data[:, :, 1]
    φᶠᶠᵃ = pFF.data[:, :, 1]

    λᶠᶜᵃ = lFC.data[:, :, 1]
    φᶠᶜᵃ = pFC.data[:, :, 1]

    λᶜᶠᵃ = lCF.data[:, :, 1]
    φᶜᶠᵃ = pCF.data[:, :, 1]

    λᶜᶜᵃ = lCC.data[:, :, 1]
    φᶜᶜᵃ = pCC.data[:, :, 1]

    # Metrics
    Δxᶜᶜᵃ = zeros(Nx, Ny)
    Δxᶠᶜᵃ = zeros(Nx, Ny)
    Δxᶜᶠᵃ = zeros(Nx, Ny)
    Δxᶠᶠᵃ = zeros(Nx, Ny)

    Δyᶜᶜᵃ = zeros(Nx, Ny)
    Δyᶠᶜᵃ = zeros(Nx, Ny)
    Δyᶜᶠᵃ = zeros(Nx, Ny)
    Δyᶠᶠᵃ = zeros(Nx, Ny)

    Azᶜᶜᵃ = zeros(Nx, Ny)
    Azᶠᶜᵃ = zeros(Nx, Ny)
    Azᶜᶠᵃ = zeros(Nx, Ny)
    Azᶠᶠᵃ = zeros(Nx, Ny)

    loop! = _calculate_metrics!(device(CPU()), (16, 16), (Nx, Ny))

    loop!(Δxᶠᶜᵃ, Δxᶜᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
          Δyᶠᶜᵃ, Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶠᵃ,
          Azᶠᶜᵃ, Azᶜᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ,
          λᶠᶜᵃ, λᶜᶜᵃ, λᶜᶠᵃ, λᶠᶠᵃ,
          φᶠᶜᵃ, φᶜᶜᵃ, φᶜᶠᵃ, φᶠᶠᵃ,
          radius)

    # Metrics fields to fill halos
    FF = Field((Face, Face, Center),     grid; boundary_conditions = default_boundary_conditions)
    FC = Field((Face, Center, Center),   grid; boundary_conditions = default_boundary_conditions)
    CF = Field((Center, Face, Center),   grid; boundary_conditions = default_boundary_conditions)
    CC = Field((Center, Center, Center), grid; boundary_conditions = default_boundary_conditions)

    # Fill all periodic halos
    set!(FF, Δxᶠᶠᵃ); 
    set!(CF, Δxᶜᶠᵃ); 
    set!(FC, Δxᶠᶜᵃ); 
    set!(CC, Δxᶜᶜᵃ); 
    fill_halo_regions!(FF)
    fill_halo_regions!(CF)
    fill_halo_regions!(FC)
    fill_halo_regions!(CC)
    Δxᶠᶠᵃ = FF.data[:, :, 1]; 
    Δxᶜᶠᵃ = CF.data[:, :, 1]; 
    Δxᶠᶜᵃ = FC.data[:, :, 1]; 
    Δxᶜᶜᵃ = CC.data[:, :, 1]; 

    set!(FF, Δyᶠᶠᵃ); set!(CF, Δyᶜᶠᵃ); set!(FC, Δyᶠᶜᵃ); set!(CC, Δyᶜᶜᵃ); 
    fill_halo_regions!(FF)
    fill_halo_regions!(CF)
    fill_halo_regions!(FC)
    fill_halo_regions!(CC)
    Δyᶠᶠᵃ = FF.data[:, :, 1]; 
    Δyᶜᶠᵃ = CF.data[:, :, 1]; 
    Δyᶠᶜᵃ = FC.data[:, :, 1]; 
    Δyᶜᶜᵃ = CC.data[:, :, 1]; 

    set!(FF, Azᶠᶠᵃ); set!(CF, Azᶜᶠᵃ); set!(FC, Azᶠᶜᵃ); set!(CC, Azᶜᶜᵃ); 
    fill_halo_regions!(FF)
    fill_halo_regions!(CF)
    fill_halo_regions!(FC)
    fill_halo_regions!(CC)
    Azᶠᶠᵃ = FF.data[:, :, 1]; 
    Azᶜᶠᵃ = CF.data[:, :, 1]; 
    Azᶠᶜᵃ = FC.data[:, :, 1]; 
    Azᶜᶜᵃ = CC.data[:, :, 1]; 

    Hx, Hy, Hz = halo

    latitude_longitude_grid = LatitudeLongitudeGrid(; size, 
                                                      latitude, 
                                                      longitude, 
                                                      z,
                                                      halo, 
                                                      radius)

    continue_south!(Δxᶠᶠᵃ, latitude_longitude_grid.Δxᶠᶠᵃ)
    continue_south!(Δxᶠᶜᵃ, latitude_longitude_grid.Δxᶠᶜᵃ)
    continue_south!(Δxᶜᶠᵃ, latitude_longitude_grid.Δxᶜᶠᵃ)
    continue_south!(Δxᶜᶜᵃ, latitude_longitude_grid.Δxᶜᶜᵃ)
    
    continue_south!(Δyᶠᶠᵃ, latitude_longitude_grid.Δyᶠᶜᵃ)
    continue_south!(Δyᶠᶜᵃ, latitude_longitude_grid.Δyᶠᶜᵃ)
    continue_south!(Δyᶜᶠᵃ, latitude_longitude_grid.Δyᶜᶠᵃ)
    continue_south!(Δyᶜᶜᵃ, latitude_longitude_grid.Δyᶜᶠᵃ)

    continue_south!(Azᶠᶠᵃ, latitude_longitude_grid.Azᶠᶠᵃ)
    continue_south!(Azᶠᶜᵃ, latitude_longitude_grid.Azᶠᶜᵃ)
    continue_south!(Azᶜᶠᵃ, latitude_longitude_grid.Azᶜᶠᵃ)
    continue_south!(Azᶜᶜᵃ, latitude_longitude_grid.Azᶜᶜᵃ)

    # Final grid with correct metrics
    grid = OrthogonalSphericalShellGrid{Periodic, RightConnected, Bounded}(arch,
                    Nx, Ny, Nz,
                    Hx, Hy, Hz,
                    convert(eltype(radius), Lz),
                    on_architecture(arch,  λᶜᶜᵃ), on_architecture(arch,  λᶠᶜᵃ), on_architecture(arch,  λᶜᶠᵃ), on_architecture(arch,  λᶠᶠᵃ),
                    on_architecture(arch,  φᶜᶜᵃ), on_architecture(arch,  φᶠᶜᵃ), on_architecture(arch,  φᶜᶠᵃ), on_architecture(arch,  φᶠᶠᵃ), on_architecture(arch, zᵃᵃᶜ),  on_architecture(arch, zᵃᵃᶠ),
                    on_architecture(arch, Δxᶜᶜᵃ), on_architecture(arch, Δxᶠᶜᵃ), on_architecture(arch, Δxᶜᶠᵃ), on_architecture(arch, Δxᶠᶠᵃ),
                    on_architecture(arch, Δyᶜᶜᵃ), on_architecture(arch, Δyᶜᶠᵃ), on_architecture(arch, Δyᶠᶜᵃ), on_architecture(arch, Δyᶠᶠᵃ), on_architecture(arch, Δzᵃᵃᶜ), on_architecture(arch, Δzᵃᵃᶠ),
                    on_architecture(arch, Azᶜᶜᵃ), on_architecture(arch, Azᶠᶜᵃ), on_architecture(arch, Azᶜᶠᵃ), on_architecture(arch, Azᶠᶠᵃ),
                    radius, Tripolar())
             
    return grid
end

function continue_south!(new_metric, lat_lon_metric::Number)
    Hx, Hy = new_metric.offsets
    Nx, Ny = size(new_metric)
    for i in Hx+1:Nx+Hx, j in Hy+1:1
        new_metric[i, j] = lat_lon_metric
    end

    return nothing
end

function continue_south!(new_metric, lat_lon_metric::AbstractArray{<:Any, 1})
    Hx, Hy = new_metric.offsets
    Nx, Ny = size(new_metric)
    for i in Hx+1:Nx+Hx, j in Hy+1:1
        new_metric[i, j] = lat_lon_metric[j]
    end

    return nothing
end

function continue_south!(new_metric, lat_lon_metric::AbstractArray{<:Any, 2})
    Hx, Hy = - new_metric.offsets
    Nx, Ny = size(new_metric)
    for i in Hx+1:Nx+Hx, j in Hy+1:1
        new_metric[i, j] = lat_lon_metric[i, j]
    end

    return nothing
end

const TRG = Union{TripolarGrid, ImmersedBoundaryGrid{<:Any, <:Any, <:Any, <:Any, <:TripolarGrid}}

import Oceananigans.Grids: x_domain, y_domain

x_domain(grid::TRG) = CUDA.@allowscalar 0, 360
y_domain(grid::TRG) = CUDA.@allowscalar minimum(grid.φᶠᶠᵃ), 90

import Oceananigans.BoundaryConditions: regularize_field_boundary_conditions
using Oceananigans.BoundaryConditions: FieldBoundaryConditions, 
                                       assumed_field_location, 
                                       regularize_boundary_condition,
                                       regularize_immersed_boundary_condition,
                                       LeftBoundary,
                                       RightBoundary

function regularize_field_boundary_conditions(bcs::FieldBoundaryConditions,
                                              grid::TRG,
                                              field_name::Symbol,
                                              prognostic_names=nothing)

    loc = assumed_field_location(field_name)

    sign = (field_name == :u || field_name == :v) ? -1 : 1

    west   = regularize_boundary_condition(bcs.west,   grid, loc, 1, LeftBoundary,  prognostic_names)
    east   = regularize_boundary_condition(bcs.east,   grid, loc, 1, RightBoundary, prognostic_names)
    south  = regularize_boundary_condition(bcs.south,  grid, loc, 2, LeftBoundary,  prognostic_names)
    north  = ZipperBoundaryCondition(sign)
    bottom = regularize_boundary_condition(bcs.bottom, grid, loc, 3, LeftBoundary,  prognostic_names)
    top    = regularize_boundary_condition(bcs.top,    grid, loc, 3, RightBoundary, prognostic_names)

    immersed = regularize_immersed_boundary_condition(bcs.immersed, grid, loc, field_name, prognostic_names)

    return FieldBoundaryConditions(west, east, south, north, bottom, top, immersed)
end

import Oceananigans.Fields: Field
using Oceananigans.Fields: architecture, 
                           validate_indices, 
                           validate_boundary_conditions,
                           validate_field_data, 
                           FieldBoundaryBuffers

sign(LX, LY) = 1
sign(::Type{Face},   ::Type{Face})   = -1
sign(::Type{Face},   ::Type{Center}) = -1
sign(::Type{Center}, ::Type{Face})   = -1
sign(::Type{Center}, ::Type{Center}) = 1

function Field((LX, LY, LZ)::Tuple, grid::TRG, data, old_bcs, indices::Tuple, op, status)
    arch = architecture(grid)
    indices = validate_indices(indices, (LX, LY, LZ), grid)
    validate_field_data((LX, LY, LZ), data, grid, indices)
    validate_boundary_conditions((LX, LY, LZ), grid, old_bcs)
    default_zipper = ZipperBoundaryCondition(sign(LX, LY))

    north_bc = old_bcs.north isa ZBC ? old_bcs.north : default_zipper
    
    new_bcs = FieldBoundaryConditions(; west = old_bcs.west, 
                                        east = old_bcs.east, 
                                        south = old_bcs.south,
                                        north = north_bc,
                                        top = old_bcs.top,
                                        bottom = old_bcs.bottom)

    buffers = FieldBoundaryBuffers(grid, data, new_bcs)

    return Field{LX, LY, LZ}(grid, data, new_bcs, indices, op, status, buffers)
end

import Oceananigans.Fields: tupled_fill_halo_regions!

function tupled_fill_halo_regions!(full_fields, grid::TRG, args...; kwargs...)
    for field in full_fields
        fill_halo_regions!(field, args...; kwargs...)
    end
end
