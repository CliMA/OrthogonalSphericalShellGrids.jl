
function save_grid_metrics!(filename, grid::TripolarGrid)
    (; λᶜᶜᵃ, φᶜᶜᵃ, λᶜᶠᵃ, φᶜᶠᵃ, λᶠᶜᵃ, φᶠᶜᵃ, λᶠᶠᵃ, φᶠᶠᵃ, Δxᶜᶜᵃ, Δyᶜᶜᵃ, Δxᶜᶠᵃ, Δyᶜᶠᵃ, Δxᶠᶜᵃ, Δyᶠᶜᵃ, Δxᶠᶠᵃ, Δyᶠᶠᵃ, Azᶜᶜᵃ, Azᶜᶠᵃ, Azᶠᶜᵃ, Azᶠᶠᵃ) = grid
    @save filename λᶜᶜᵃ φᶜᶜᵃ λᶜᶠᵃ φᶜᶠᵃ λᶠᶜᵃ φᶠᶜᵃ λᶠᶠᵃ φᶠᶠᵃ Δxᶜᶜᵃ Δyᶜᶜᵃ Δxᶜᶠᵃ Δyᶜᶠᵃ Δxᶠᶜᵃ Δyᶠᶜᵃ Δxᶠᶠᵃ Δyᶠᶠᵃ Azᶜᶜᵃ Azᶜᶠᵃ Azᶠᶜᵃ Azᶠᶠᵃ

    return nothing
end

function TripolarGrid(filename::String;
    arch = CPU(),
    halos = (5, 5, 5),
    Nz, 
    z)

    @load filename λᶜᶜᵃ φᶜᶜᵃ λᶜᶠᵃ φᶜᶠᵃ λᶠᶜᵃ φᶠᶜᵃ λᶠᶠᵃ φᶠᶠᵃ Δxᶜᶜᵃ Δyᶜᶜᵃ Δxᶜᶠᵃ Δyᶜᶠᵃ Δxᶠᶜᵃ Δyᶠᶜᵃ Δxᶠᶠᵃ Δyᶠᶠᵃ Azᶜᶜᵃ Azᶜᶠᵃ Azᶠᶜᵃ Azᶠᶠᵃ

    TX, TY, TZ = (Periodic, RightConnected, Bounded)
    Hx, Hy, Hz = halos

    Nx, Ny = size(λᶜᶜᵃ) .- 2 .* (Hx, Hy)

    z_grid = RectilinearGrid(arch; size = Nz, halo = halos[3], z, topology = (Flat, Flat, Bounded))

    grid = OrthogonalSphericalShellGrid{TX, TY, TZ}(CPU(),
                                                    Nx, Ny, Nz,
                                                    Hx, Hy, Hz,
                                                    z_grid.Lz,
                                                    λᶜᶜᵃ,  λᶠᶜᵃ,  λᶜᶠᵃ,  λᶠᶠᵃ,
                                                    φᶜᶜᵃ,  φᶠᶜᵃ,  φᶜᶠᵃ,  φᶠᶠᵃ, z_grid.zᵃᵃᶜ, z_grid.zᵃᵃᶠ,
                                                    Δxᶜᶜᵃ, Δxᶠᶜᵃ, Δxᶜᶠᵃ, Δxᶠᶠᵃ,
                                                    Δyᶜᶜᵃ, Δyᶜᶠᵃ, Δyᶠᶜᵃ, Δyᶠᶠᵃ, z_grid.Δzᵃᵃᶜ, z_grid.Δzᵃᵃᶠ,
                                                    Azᶜᶜᵃ, Azᶠᶜᵃ, Azᶜᶠᵃ, Azᶠᶠᵃ,
                                                    R_Earth,
                                                    TripolarMapping())

    return on_architecture(arch, grid)
end