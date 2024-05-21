#
using KolmogorovArnold
using Random, LinearAlgebra

# Add test dependencies to env stack
let 
    pkgpath = dirname(dirname(pathof(KolmogorovArnold)))
    tstpath = joinpath(pkgpath, "test")
    !(tstpath in LOAD_PATH) && push!(LOAD_PATH, tstpath)
end

using Zygote, Lux, ComponentArrays
using LuxDeviceUtils, CUDA, LuxCUDA
using BenchmarkTools

# configure BLAS
ncores = min(Sys.CPU_THREADS, length(Sys.cpu_info()))
BLAS.set_num_threads(ncores)

# configure CUDA
CUDA.allowscalar(false)

rng = Random.default_rng()
Random.seed!(rng, 0)
device = Lux.gpu_device()

#======================================================#
function main()
    x = rand32(rng, 1, 1000) |> device

    nM, nK, G =  32, 10, 10
    nM, nK, G = 128, 40, 10

    mlp = Chain(
        Dense(1, nM, tanh),
        Dense(nM, nM, tanh),
        Dense(nM, 1),
    )

    use_base_act = false
    basis_func = rbf # rbf, rswaf
    normalizer = softsign
    
    kan = Chain(
        KDense( 1, nK, G; use_base_act, basis_func, normalizer),
        KDense(nK, nK, G; use_base_act, basis_func, normalizer),
        KDense(nK,  1, G; use_base_act, basis_func, normalizer),
    )

    display(mlp)
    display(kan)

    pM, stM = Lux.setup(rng, mlp)
    pK, stK = Lux.setup(rng, kan)

    pM = ComponentArray(pM) |> device
    pK = ComponentArray(pK) |> device

    stM, stK = device(stM), device(stK)

    f_mlp(p) = mlp(x, p, stM)[1] |> sum
    f_kan(p) = kan(x, p, stK)[1] |> sum

    # # Zygote is type unstable - consider using generated functinos
    # _, pbM = Zygote.pullback(f_mlp, pM)
    # _, pbK = Zygote.pullback(f_kan, pK)

    # @code_warntype pbM(x)
    # @code_warntype pbK(x)

    if device isa LuxDeviceUtils.AbstractLuxGPUDevice
        println("# FWD PASS")
    
        @btime CUDA.@sync $mlp($x, $pM, $stM)
        @btime CUDA.@sync $kan($x, $pK, $stK)
    
        println("# BWD PASS")
    
        @btime CUDA.@sync Zygote.gradient($f_mlp, $pM)
        @btime CUDA.@sync Zygote.gradient($f_kan, $pK)
    else
        println("# FWD PASS")
    
        @btime $mlp($x, $pM, $stM)
        @btime $kan($x, $pK, $stK)
    
        println("# BWD PASS")
    
        @btime Zygote.gradient($f_mlp, $pM)
        @btime Zygote.gradient($f_kan, $pK)
    end

    nothing
end
#======================================================#
main()
#
