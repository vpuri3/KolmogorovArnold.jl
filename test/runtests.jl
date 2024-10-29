using KolmogorovArnold
using Test


pkgpath = dirname(dirname(pathof(KolmogorovArnold)))

# Write your tests here.


@testset "FunctionFit" begin
    include(joinpath(pkgpath, "examples", "eg4.jl"))
end

@testset "Speedtest" begin
    include(joinpath(pkgpath, "examples", "eg1.jl"))
end

