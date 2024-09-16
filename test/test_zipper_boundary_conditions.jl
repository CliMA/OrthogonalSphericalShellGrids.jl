include("dependencies_for_runtests.jl")

using OrthogonalSphericalShellGrids: Zipper

@testset "Zipper boundary conditions..." begin
    grid = TripolarGrid(size = (10, 10, 1))
    Nx, Ny, _ = size(grid)
    Hx, Hy, _ = halo_size(grid)

    c = CenterField(grid)
    u = XFaceField(grid)
    v = YFaceField(grid)

    @test c.boundary_conditions.north.classification isa Zipper
    @test u.boundary_conditions.north.classification isa Zipper
    @test v.boundary_conditions.north.classification isa Zipper

    @test c.boundary_conditions.north.condition == 1
    @test u.boundary_conditions.north.condition == -1
    @test v.boundary_conditions.north.condition == -1

    set!(c, 1.0)
    set!(u, 1.0)
    set!(v, 1.0)

    fill_halo_regions!(c)
    fill_halo_regions!(u)   
    fill_halo_regions!(v)

    north_boundary_c = view(c.data, :, Ny+1:Ny+Hy, 1)
    north_boundary_v = view(v.data, :, Ny+1:Ny+Hy, 1)
    @test all(north_boundary_c .== 1.0)
    @test all(north_boundary_v .== -1.0)

    # U is special, because periodicity is hardcoded in the x-direction
    north_interior_boundary_u = view(u.data, 2:Nx-1, Ny+1:Ny+Hy, 1)
    @test all(north_interior_boundary_u .== -1.0)

    north_boundary_u_left  = view(u.data, 1, Ny+1:Ny+Hy, 1)
    north_boundary_u_right = view(u.data, Nx+1, Ny+1:Ny+Hy, 1)
    @test all(north_boundary_u_left  .== 1.0)
    @test all(north_boundary_u_right .== 1.0)
end