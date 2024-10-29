#
using KolmogorovArnold

# Add test dependencies to env stack
let 
    pkgpath = dirname(dirname(pathof(KolmogorovArnold)))
    tstpath = joinpath(pkgpath, "test")
    !(tstpath in LOAD_PATH) && push!(LOAD_PATH, tstpath)
end

using Random, LinearAlgebra
using Plots, NNlib, Zygote, BenchmarkTools
using MLDataDevices, CUDA, LuxCUDA

# configure BLAS
ncores = min(Sys.CPU_THREADS, length(Sys.cpu_info()))
BLAS.set_num_threads(ncores)

# configure CUDA
CUDA.allowscalar(false)

rng = Random.default_rng()
Random.seed!(rng, 0)
device = gpu_device()

#======================================================#

sigmoid_norm(x) = 2 * sigmoid(x) - 1

function curve(
    x::AbstractVector, # [N]
    c::AbstractVector, # [G]
    z::AbstractVector, # [G]
    d::Number,
    basis_func,
)
    vec(c' * basis_func(x', z, d))
end

function plot_basis(N = 1000, G = 10)
    x = LinRange(-1, 1, N)
    z = LinRange(-1, 1, G)
    d = 2 / (G-1)

    # y_rbf = rbf.(x, z', d)
    # y_rswaf = rswaf.(x, z', d)
    
    i = LinRange(1, N, G) .|> Base.Fix1(round, Int)
    f = sin.(4pi * x)
    c = f[i]

    y_rbf   = curve(x, c, z, d, rbf)
    y_rswaf = curve(x, c, z, d, rswaf)
    y_iqf   = curve(x, c, z, d, iqf)

    plt = plot(
        xlabel = "x",
        ylabel = "basis functions",
        title = "Are the basis expressive enough?"
    )

    plot!(plt, x, y_rbf  ; w = 3, c = :blue , label = "RBF")
    plot!(plt, x, y_rswaf; w = 3, c = :red  , label = "RSWAF")
    plot!(plt, x, y_iqf  ; w = 3, c = :green, label = "IQF")

    plt
end

function plot_normalizers(N = 1000)
    x = LinRange(-5, 5, N)

    plt = plot(
        xlabel = "x",
        ylabel = "Normalizers",
        title = "What's the most unobtrusive normalizer?"
    )

    plot!(plt, x, tanh        ; w = 3, c = :blue , label = "tanh")
    plot!(plt, x, sigmoid_norm; w = 3, c = :red  , label = "sigmoid")
    plot!(plt, x, softsign    ; w = 3, c = :green, label = "softsign")

    return plt
end

function speedtest(N = 5000, G = 10)
    x = LinRange(-1, 1, N) |> Array |> device
    z = LinRange(-1, 1, G) |> Array |> device
    d = 2 / (G-1)

    f_rbf(z)   = rbf(  x, z', d) |> sum
    f_rswaf(z) = rswaf(x, z', d) |> sum
    f_iqf(z)   = iqf(  x, z', d) |> sum

    println("# FWD PASS")
    @btime CUDA.@sync $f_rbf($z)  
    @btime CUDA.@sync $f_rswaf($z)
    @btime CUDA.@sync $f_iqf($z)  

    println("# BWD PASS")
    @btime CUDA.@sync Zygote.gradient($f_rbf  , $z)
    @btime CUDA.@sync Zygote.gradient($f_rswaf, $z)
    @btime CUDA.@sync Zygote.gradient($f_iqf  , $z)
end

#======================================================#
p1 = plot_basis()
p2 = plot_normalizers()
speedtest()
nothing
#
