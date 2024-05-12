using KolmogorovArnold
using Test

using Lux, Zygote
using Optimisers, OptimizationOptimJL

pkgpath = dirname(dirname(pathof(KolmogorovArnold)))

@testset "KolmogorovArnold.jl" begin
    # Write your tests here.
    include(joinpath(pkgpath, "examples", "eg1.jl"))
end
