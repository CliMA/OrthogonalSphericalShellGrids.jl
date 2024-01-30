using Documenter
using OrthogonalSphericalShellGrids

api = Any[
    "API" => "grids.md"
]

makedocs(
    sitename = "WenoNeverworld",
    format = Documenter.HTML(),
#     modules = [WenoNeverworld]
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
deploydocs(repo = "github.com/simone-silvestri/WenoNeverworld.jl.git", push_preview = true)
