using Documenter, Ket, MATLAB

makedocs(
    sitename="Ket",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true")
    )
