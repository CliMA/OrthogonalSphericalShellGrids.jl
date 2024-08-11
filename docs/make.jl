using Documenter
using OrthogonalSphericalShellGrids
using Literate

using GLMakie # to avoid capturing precompilation output by Literate

#####
##### Generate examples
#####

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR   = joinpath(@__DIR__, "src/literated")

example_scripts = [
    "visualize_tripolar_grid.jl",
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
    "Examples" => [
        "Tripolar Grid" => "literated/visualize_tripolar_grid.md",
        # "Bickley jet" => "literated/bickley_jet.md",
    ],
"API" => "API.md",
]

#####
##### Build and deploy docs
#####

format = Documenter.HTML(collapselevel = 1,
                         mathengine = MathJax3(),
                         size_threshold = 3 * 1024^2)

makedocs(sitename = "OrthogonalSphericalShellGrids.jl",
         authors = "CliMA",
         pages = pages,
         format = format,
         doctest = true, # set to false to speed things up
         clean = true,
         checkdocs = :exports) # set to :none to speed things up

deploydocs(repo = "github.com/CliMA/OrthogonalSphericalShellGrids.jl.git",
           forcepush = true,
           push_preview = true,
           devbranch = "main")
