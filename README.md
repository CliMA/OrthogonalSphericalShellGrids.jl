<!-- Title -->
<h1 align="center">
  OrthogonalSphericalShellGrids.jl
</h1>

<!-- description -->
<p align="center">
  <strong>🌐 Recipes and tools for Tools for constructing  <a href="https://github.com/CliMA/Oceananigans.jl">Oceananigans</a> grids that represent orthogonal meshes of thin spherical shells, which prove particularly useful for ocean simulations.</strong>
</p>

<!-- Information badges -->
<p align="center">
    <a href="https://mit-license.org">
        <img alt="MIT license" src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square">
    </a>
    <a href="https://clima.github.io/OrthogonalSphericalShellGrids.jl/stable">
        <img alt="Documentation" src="https://img.shields.io/badge/documentation-stable%20release-blue?style=flat-square">
    </a>
    <a href="https://clima.github.io/OrthogonalSphericalShellGrids.jl/dev">
        <img alt="Documentation" src="https://img.shields.io/badge/documentation-in%20development-orange?style=flat-square">
    </a>
    <a href="https://github.com/CliMA/OrthogonalSphericalShellGrids.jl/actions/workflows/CI.yml?query=branch%3Amain">
        <img alt="Build status" src="https://github.com/simone-silvestri/OrthogonalSphericalShellGrids.jl/actions/workflows/CI.yml/badge.svg?branch=main">
    </a>
</p>


OrthogonalSphericalShellGrids includes tools for generating spherical shell grids for ocean simulations with Oceananigans.

OrthogonalSphericalShellGrids is developed by the [Climate Modeling Alliance](https://clima.caltech.edu).

## Quick install

OrthogonalSphericalShellGrids is a [registered Julia package](https://julialang.org/packages/). So to install it,

1. [Download Julia](https://julialang.org/downloads/).

2. Launch Julia and type

```julia
julia> using Pkg

julia> Pkg.add("OrthogonalSphericalShellGrids")
```

The example [`examples/visualize_tripolar_grid.jl`](https://clima.github.io/OrthogonalSphericalShellGrids.jl/dev/literated/visualize_tripolar_grid/) generates a visualization of the `TripolarGrid` ([generated by a series of cofocal ellipses perpendicular to a family of hyperbolae](https://www.sciencedirect.com/science/article/abs/pii/S0021999196901369)),
producing

<p align="center">
<img width="1000" alt="tripolar grid visualization" src="https://github.com/user-attachments/assets/c2ffa8b0-fc40-4ec4-9eaa-d722ba001435">

A tripolar grid. The North pole singularity (left) has been "healed" by having two poles
at 60 degrees latitude. The South pole (right) retained its singularity. This grid is
particularly useful in ocean modeling so that we avoid having to deal with numerical issues
arising from the North pole singularity within the Arctic ocean.
</p>
