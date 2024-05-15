using Documenter
using OrthogonalSphericalShellGrids

api = Any[
    "API" => "grids.md"
]

makedocs(
    sitename = "OrthogonalSphericalShellGrids",
    format = Documenter.HTML(),
    pages=[
        "Home" => "index.md",
        "API" => api,
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#


# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(repo = "github.com/simone-silvestri/OrthogonalSphericalShellGrids.jl.git", push_preview = true)
