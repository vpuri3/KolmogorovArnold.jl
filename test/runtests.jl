using KolmogorovArnold
using Test

pkgpath = dirname(dirname(pathof(KolmogorovArnold)))

@testset "KolmogorovArnold.jl" begin
    # Write your tests here.
    include(joinpath(pkgpath, "examples", "eg1.jl"))
end
