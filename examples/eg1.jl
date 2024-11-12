
using KolmogorovArnold

# Add test dependencies to env stack
let 
    pkgpath = dirname(dirname(pathof(KolmogorovArnold)))
    tstpath = joinpath(pkgpath, "test")
    !(tstpath in LOAD_PATH) && push!(LOAD_PATH, tstpath)
end

using Random, LinearAlgebra
using Zygote, Lux, ComponentArrays
using MLDataDevices, CUDA, LuxCUDA
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
function main(N=1000)
    x  = rand32(rng, 1, N) |> device
    x₀ = rand32(rng, N, 1) |> device

    wM, wK, wK2, G = 128, 40, 30, 10 # MLP width, KAN width, grid size

    mlp = Chain(
        Dense(1, wM, tanh),
        Dense(wM, wM, tanh),
        Dense(wM, 1),
    )

    basis_func = rbf      # rbf, rswaf
    normalizer = softsign # sigmoid(_fast), tanh(_fast), softsign
    
    kan1 = Chain(
        KDense( 1, wK, G; use_base_act = true, basis_func, normalizer),
        KDense(wK, wK, G; use_base_act = true, basis_func, normalizer),
        KDense(wK,  1, G; use_base_act = true, basis_func, normalizer),
    )

    kan2 = Chain(
        KDense( 1, wK, G; use_base_act = false, basis_func, normalizer),
        KDense(wK, wK, G; use_base_act = false, basis_func, normalizer),
        KDense(wK,  1, G; use_base_act = false, basis_func, normalizer),
    )

    kan3 = Chain(
        CDense( 1, wK, G),
        CDense(wK, wK, G),
        CDense(wK,  1, G),
    )

    kan4 = Chain(
        FDense( 1, wK2, G),
        FDense(wK2, wK2, G),
        FDense(wK2,  1, G),
    )

    display(mlp)
    display(kan1)
    display(kan2)
    display(kan3)
    display(kan4)

    pM, stM = Lux.setup(rng, mlp)
    pK1, stK1 = Lux.setup(rng, kan1)
    pK2, stK2 = Lux.setup(rng, kan2)
    pK3, stK3 = Lux.setup(rng, kan3)
    pK4, stK4 = Lux.setup(rng, kan4)


    pM  = ComponentArray(pM) |> device
    pK1 = ComponentArray(pK1) |> device
    pK2 = ComponentArray(pK2) |> device
    pK3 = ComponentArray(pK3) |> device
    pK4 = ComponentArray(pK4) |> device

    stM, stK1, stK2, stK3, stK4 = device(stM), device(stK1), device(stK2), device(stK4), device(stK4)

    f_mlp(p)  = mlp(x, p, stM)[1] |> sum
    f_kan1(p) = kan1(x, p, stK1)[1] |> sum
    f_kan2(p) = kan2(x, p, stK2)[1] |> sum
    f_kan3(p) = kan3(x₀, p, stK3)[1] |> sum
    f_kan4(p) = kan4(x₀, p, stK4)[1] |> sum

    # # Zygote is type unstable - consider using generated functinos
    # _, pbM = Zygote.pullback(f_mlp, pM)
    # _, pbK1 = Zygote.pullback(f_kan1, pK)
    # _, pbK2 = Zygote.pullback(f_kan2, pK)

    # @code_warntype pbM(x)
    # @code_warntype pbK(x)

    if device isa MLDataDevices.AbstractGPUDevice
        println("# FWD PASS")
    
        @btime CUDA.@sync $mlp($x, $pM, $stM)
        @btime CUDA.@sync $kan1($x, $pK1, $stK1)
        @btime CUDA.@sync $kan2($x, $pK2, $stK2)
        @btime CUDA.@sync $kan3($x₀, $pK3, $stK3)
        @btime CUDA.@sync $kan4($x₀, $pK4, $stK4)

        println("# BWD PASS")

        @btime CUDA.@sync Zygote.gradient($f_mlp, $pM)
        @btime CUDA.@sync Zygote.gradient($f_kan1, $pK1)
        @btime CUDA.@sync Zygote.gradient($f_kan2, $pK2)
        @btime CUDA.@sync Zygote.gradient($f_kan3, $pK3)
        @btime CUDA.@sync Zygote.gradient($f_kan4, $pK4)
    else
        println("# FWD PASS")

        @btime $mlp($x, $pM, $stM)
        @btime $kan1($x, $pK1, $stK1)
        @btime $kan2($x, $pK2, $stK2)
        @btime $kan3($x₀, $pK3, $stK3)
        @btime $kan4($x₀, $pK4, $stK4) 

        println("# BWD PASS")

        @btime Zygote.gradient($f_mlp, $pM)
        @btime Zygote.gradient($f_kan1, $pK1)
        @btime Zygote.gradient($f_kan2, $pK2)
        @btime Zygote.gradient($f_kan3, $pK3)
        @btime Zygote.gradient($f_kan4, $pK4)
    end

    nothing
end

#======================================================#
main()
#
