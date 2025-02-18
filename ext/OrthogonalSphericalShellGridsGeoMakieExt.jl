module OrthogonalSphericalShellGridsGeoMakieExt

using GeoMakie
using Oceananigans
using Oceananigans.Utils
using Oceananigans.Grids: λnode, φnode, on_architecture, AbstractGrid
using GeoMakie: GeometryBasics

using KernelAbstractions: @kernel, @index

function globe(data::Observable, grid; add_coastlines=true, colormap=:viridis, colorrange=nothing, nan_color=:white)
    fig = Figure(size=(800, 800));

    ax = LScene(fig[1,1], show_axis=false);
    transf = GeoMakie.Geodesy.ECEFfromLLA(GeoMakie.Geodesy.WGS84())

    faces = list_cell_vertices(grid; add_nans=false)

    polygons = [GeometryBasics.Polygon(faces[:, j]) for j in 1:length(data[])]
    if isnothing(colorrange)
        p = poly!(ax, polygons, color=data; colormap, nan_color)
    else
        p = poly!(ax, polygons, color=data; colormap, colorrange, nan_color)
    end
    p.transformation.transform_func[] = transf

    if add_coastlines
        c = lines!(GeoMakie.coastlines(50); color=:white, linewidth=1, alpha=0.7)
        c.transformation.transform_func[] = transf
    end

    cc = cameracontrols(ax.scene)
    cc.settings.mouse_translationspeed[] = 0.0
    cc.settings.zoom_shift_lookat[] = false
    cc.eyeposition[] = Float32[3.0, 3.0, 3.0]
    Makie.update_cam!(ax.scene, cc)
    
    rotate_cam!(ax.scene, (- π/3.5, 0, 0))
    
    return fig, ax
end

function globe(field::Field; add_coastlines=true, colormap=:viridis, level=size(field, 3), colorrange=nothing, nan_color=:white)
    fig = Figure(size=(800, 800));

    ax = LScene(fig[1,1], show_axis=false);
    transf = GeoMakie.Geodesy.ECEFfromLLA(GeoMakie.Geodesy.WGS84())

    grid  = field.grid
    data  = on_architecture(Oceananigans.CPU(), vec(interior(field, :, :, level))[1:end])
    faces = list_cell_vertices(grid; add_nans=false)

    polygons = [GeometryBasics.Polygon(faces[:, j]) for j in 1:length(data)]
    if isnothing(colorrange)
        p = poly!(ax, polygons, color=data; colormap, nan_color)
    else
        p = poly!(ax, polygons, color=data; colormap, colorrange, nan_color)
    end
    p.transformation.transform_func[] = transf

    if add_coastlines
        c = lines!(GeoMakie.coastlines(50); color=:white, linewidth=1, alpha=0.7)
        c.transformation.transform_func[] = transf
    end

    cc = cameracontrols(ax.scene)
    cc.settings.mouse_translationspeed[] = 0.0
    cc.settings.zoom_shift_lookat[] = false
    Makie.update_cam!(ax.scene, cc)
    
    return fig
end

function globe(grid::AbstractGrid; add_coastlines=true, color=:black)
    fig = Figure(size=(800, 800));

    ax = LScene(fig[1,1], show_axis=false);
    transf = GeoMakie.Geodesy.ECEFfromLLA(GeoMakie.Geodesy.WGS84())

    bg = meshimage!(ax, -180..180, -90..90, rotr90(GeoMakie.earth()); npoints = 100, z_level = -10_000);
    bg.transformation.transform_func[] = transf

    if add_coastlines
        cl = lines!(GeoMakie.coastlines(50); color=:black, linewidth=1)
        cl.transformation.transform_func[] = transf
    end

    faces = list_cell_vertices(grid)
    # add nan after every face to avoid lines linking grid cells
    f = lines!(ax, vec(faces); color)
    f.transformation.transform_func[] = transf

    cc = cameracontrols(ax.scene)
    cc.settings.mouse_translationspeed[] = 0.0
    cc.settings.zoom_shift_lookat[] = false
    Makie.update_cam!(ax.scene, cc)
    
    return fig
end

"""
    list_cell_vertices(grid)

Returns a list representing all horizontal grid cells in a curvilinear `grid`. 
The outpur is an Array of 6 * M `Point2` elements where `M = Nx * Ny`. Each row lists the vertices associated with a
horizontal cell in clockwise order starting from the southwest (bottom left) corner.
"""
function list_cell_vertices(grid; add_nans=true)
    Nx, Ny, _ = size(grid)
    FT = eltype(grid)

    cpu_grid = on_architecture(Oceananigans.CPU(), grid)

    sw  = fill(Point2{FT}(0, 0),     1, Nx*Ny)
    nw  = fill(Point2{FT}(0, 0),     1, Nx*Ny)
    ne  = fill(Point2{FT}(0, 0),     1, Nx*Ny)
    se  = fill(Point2{FT}(0, 0),     1, Nx*Ny)
    nan = fill(Point2{FT}(NaN, NaN), 1, Nx*Ny)

    launch!(Oceananigans.CPU(), cpu_grid, :xy, _get_vertices!, sw, nw, ne, se, grid)
    
    vertices = vcat(sw, nw, ne, se, sw)
    
    if add_nans
        vertices = vcat(vertices, nan)
    end

    return vertices
end

@kernel function _get_vertices!(sw, nw, ne, se, grid)
    i, j = @index(Global, NTuple)

    FT  = eltype(grid)
    Nx  = size(grid, 1)
    λ⁻⁻ = λnode(i,   j,   1, grid, Face(), Face(), nothing)
    λ⁺⁻ = λnode(i,   j+1, 1, grid, Face(), Face(), nothing)
    λ⁻⁺ = λnode(i+1, j,   1, grid, Face(), Face(), nothing)
    λ⁺⁺ = λnode(i+1, j+1, 1, grid, Face(), Face(), nothing)
    
    φ⁻⁻ = φnode(i,   j,   1, grid, Face(), Face(), nothing)
    φ⁺⁻ = φnode(i,   j+1, 1, grid, Face(), Face(), nothing)
    φ⁻⁺ = φnode(i+1, j,   1, grid, Face(), Face(), nothing)
    φ⁺⁺ = φnode(i+1, j+1, 1, grid, Face(), Face(), nothing)

    sw[i+(j-1)*Nx] = Point2{FT}(λ⁻⁻, φ⁻⁻)  
    nw[i+(j-1)*Nx] = Point2{FT}(λ⁻⁺, φ⁻⁺)
    ne[i+(j-1)*Nx] = Point2{FT}(λ⁺⁺, φ⁺⁺)
    se[i+(j-1)*Nx] = Point2{FT}(λ⁺⁻, φ⁺⁻)
end

end
