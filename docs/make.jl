using AugmentedMixing
using Documenter

DocMeta.setdocmeta!(AugmentedMixing, :DocTestSetup, :(using AugmentedMixing); recursive=true)

makedocs(;
    modules=[AugmentedMixing],
    authors="Jan Schwiddessen <jan.schwiddessen@gmail.com> and contributors",
    sitename="AugmentedMixing.jl",
    format=Documenter.HTML(;
        canonical="https://jschwiddessen.github.io/AugmentedMixing.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jschwiddessen/AugmentedMixing.jl",
    devbranch="main",
)
