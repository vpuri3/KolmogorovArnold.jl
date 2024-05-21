# evaluate on MNIST1D
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
using MLUtils, MLDatasets

# configure BLAS
ncores = min(Sys.CPU_THREADS, length(Sys.cpu_info()))
BLAS.set_num_threads(ncores)

# configure CUDA
CUDA.allowscalar(false)

rng = Random.default_rng()
Random.seed!(rng, 0)
device = Lux.gpu_device()

#======================================================#
