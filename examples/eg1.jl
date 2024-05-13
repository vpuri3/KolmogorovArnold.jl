#
using Random
using KolmogorovArnold

# Add test dependencies to env stack
let 
    pkgpath = dirname(dirname(pathof(KolmogorovArnold)))
    tstpath = joinpath(pkgpath, "test")
    !(tstpath in LOAD_PATH) && push!(LOAD_PATH, tstpath)
end

using Lux, ComponentArrays
using LuxDeviceUtils, CUDA, LuxCUDA
using BenchmarkTools

rng = Random.default_rng()
Random.seed!(rng, 0)
device = Lux.gpu_device()

function main()
    x = rand32(rng, 1, 1000) |> device

    mlp = Chain(
        Dense(1, 32, tanh),
        Dense(32, 32, tanh),
        Dense(32, 1),
    )

    kan = Chain(
        KDense(1, 8, 15; use_base_act = false),
        KDense(8, 8, 15; use_base_act = false),
        KDense(8, 1, 15; use_base_act = false),
    )

    # display(mlp)
    # display(kan)

    pM, stM = Lux.setup(rng, mlp)
    pK, stK = Lux.setup(rng, kan)

    pM = ComponentArray(pM) |> device
    pK = ComponentArray(pK) |> device

    stM, stK = device(stM), device(stK)

    @btime CUDA.@sync $mlp($x, $pM, $stM)
    @btime CUDA.@sync $kan($x, $pK, $stK)

    nothing
end

main()

nothing
