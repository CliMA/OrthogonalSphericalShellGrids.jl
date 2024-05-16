using Documenter
using OrthogonalSphericalShellGrids
using Literate
using Printf

using CairoMakie # to avoid capturing precompilation output by Literate
CairoMakie.activate!(type = "svg")

#####
##### Generate examples
#####

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR   = joinpath(@__DIR__, "src/literated")

example = "generate_grid.jl"
example_filepath = joinpath(EXAMPLES_DIR, example)
withenv("JULIA_DEBUG" => "Literate") do
    start_time = time_ns()
    Literate.markdown(example_filepath, OUTPUT_DIR;
                      flavor = Literate.DocumenterFlavor(), execute = true)
    elapsed = 1e-9 * (time_ns() - start_time)
    @info @sprintf("%s example took %s to build.", example, prettytime(elapsed))
end

#####
##### Organize page hierarchies
#####

pages = [
    "Home" => "index.md",
    "API" => "grids.md"
    "Generate Grid" => "literated/generate_grid.md"
]

#####
##### Build and deploy docs
#####

format = Documenter.HTML(collapselevel = 1,
                         prettyurls = get(ENV, "CI", nothing) == "true",
                         canonical = "https://simone-silvestri.github.io/OrthogonalSphericalShellGrids/stable/",
                         mathengine = MathJax3(),
                         size_threshold = 2^20)

makedocs(sitename = "OrthogonalSphericalShellGrids.jl",
         authors = "Simone Silvestri",
         pages = pages,
         doctest = true, # set to false to speed things up
         clean = true,
         checkdocs = :exports) # set to :none to speed things up

makedocs( ; 
    sitename = "OrthogonalSphericalShellGrids",
    format = Documenter.HTML(),
    pages
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

deploydocs(repo = "github.com/simone-silvestri/OrthogonalSphericalShellGrids.jl.git",
           versions = ["stable" => "v^", "dev" => "dev", "v#.#.#"],
           forcepush = true,
           push_preview = false,
           devbranch = "main")