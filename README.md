# OrthogonalSphericalShellGrids

[![Build Status](https://github.com/simone-silvestri/OrthogonalSphericalShellGrids.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/simone-silvestri/OrthogonalSphericalShellGrids.jl/actions/workflows/CI.yml?query=branch%3Amain)

<a href="https://mit-license.org">
    <img alt="MIT license" src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square">
</a>
<a href="https://clima.github.io/OrthogonalSphericalShellGrids.jl/dev">
    <img alt="Documentation" src="https://img.shields.io/badge/documentation-stable%20release-red?style=flat-square">
</a>
 
A Tripolar grid build as a series of cofocal ellipses perpendicular to a family of hyperbolae, follows recipes from https://www.sciencedirect.com/science/article/abs/pii/S0021999196901369.

The output of `examples/generate_grid.jl`
<img width="571" alt="Screen Shot 2024-05-14 at 10 45 13 PM" src="https://github.com/simone-silvestri/OrthogonalSphericalShellGrids.jl/assets/33547697/a22d3b87-1172-4309-a26f-e0824b5a2c1a">

A sixth degree global ocean simulation using the `TripolarGrid` (left: vertical vorticity, center: surface speed, right: temperature)


