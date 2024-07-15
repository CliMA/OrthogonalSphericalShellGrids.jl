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

example_scripts = [
    "generate_grid.jl",
]

for example in example_scripts
    example_filepath = joinpath(EXAMPLES_DIR, example)
    withenv("JULIA_DEBUG" => "Literate") do
        Literate.markdown(example_filepath, OUTPUT_DIR;
                          flavor = Literate.DocumenterFlavor(), execute = true)
    end
end

#####
##### Organize page hierarchies
#####

pages = [
    "Home" => "index.md",
    "API" => "grids.md",
    "Generate Grid" => "literated/generate_grid.md"
#    "Bickley jet" => "literated/bickley_jet.md"
]

#####
##### Build and deploy docs
#####

format = Documenter.HTML(collapselevel = 1,
                         mathengine = MathJax3(),
                         size_threshold = 3 * 1024^2)

makedocs(sitename = "OrthogonalSphericalShellGrids.jl",
         authors = "Simone Silvestri",
         pages = pages,
         format = format,
         doctest = true, # set to false to speed things up
         clean = true,
         checkdocs = :exports) # set to :none to speed things up

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
           forcepush = true,
           push_preview = true,
           devbranch = "main")