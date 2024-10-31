using KolmogorovArnold
using Test, cuTENSOR

pkgpath = dirname(dirname(pathof(KolmogorovArnold)))

# Write your tests here.

@testset "Speedtest" begin
    include(joinpath(pkgpath, "examples", "eg1.jl"))
end


@testset "FunctionFit" begin
    include(joinpath(pkgpath, "examples", "eg4.jl"))
end

