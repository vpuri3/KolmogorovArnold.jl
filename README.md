# KolmogorovArnold.jl

[![Build Status](https://github.com/vpuri3/KolmogorovArnold.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/vpuri3/KolmogorovArnold.jl/actions/workflows/CI.yml?query=branch%3Amaster)

Julia implementation of the [Kolmogorov-Arnold network](https://arxiv.org/abs/2404.19756)
for the [`Lux.jl`](https://lux.csail.mit.edu/stable/) framework.
This implementation is based on [efficient-kan](https://github.com/Blealtan/efficient-kan)
and ['FastKAN'](https://github.com/ZiyaoLi/fast-kan) which resolve the performance
issues with the [original implementation](https://github.com/KindXiaoming/pykan).
Key implementation details here are:
- We fix our grid to be in `[-1, 1]` and normalize the the input to lie in that interval with `tanh` or `NNlib.tanh_fast`.
- We use radial basis functions in place of the spline basis as the former is very efficient to evaluate.

```julia
using Random, KolmogorovArnold
rng = Random.default_rng()

in_dim, out_dim, grid_len = 4, 4, 8
layer = KDense(in_dim, out_dim, grid_len)
p, st = Lux.setup(rng, layer)

x = rand32(rng, in_dim, 10)
y = layer(x, p, st)
```

We compare the performance of KAN with an MLP that has the same number of parameters.
```julia
using Lux, KolmogorovArnold
using CUDA, LuxDeviceUtils
device = Lux.gpu_device()

mlp = Chain(Dense(1, 32, tanh), Dense(32, 32, tanh), Dense(32, 1),) # 1_153 parameters
kan = Chain(KDense(1, 10, 10), KDense(10, 10, 10), KDense(10, 1, 10)) # 1_320 parameters plus 30 states

x = rand32(rng, 1, 1000) |> device
pM, stM = Lux.setup(rng, mlp) |> device
pK, stK = Lux.setup(rng, kan) |> device

# Forward pass
@btime CUDA.@sync $mlp($x, $pM, $stM) # 34.360 μs (175 allocations: 4.78 KiB)
@btime CUDA.@sync $kan($x, $pK, $stK) # 155.781 μs (565 allocations: 17.50 KiB)

f_mlp(p) = mlp(x, p, stM)[1] |> sum
f_kan(p) = kan(x, p, stK)[1] |> sum

# Backward pass
@btime CUDA.@sync Zygote.gradient($f_mlp, $pM) # 446.835 μs (1498 allocations: 56.94 KiB)
@btime CUDA.@sync Zygote.gradient($f_kan, $pK) # 1.250 ms (3879 allocations: 136.06 KiB)

```
With `use_base_act = false`, the performance of KAN effectively doubles
```julia
kan = Chain(
    KDense( 1, 10, 10; use_base_act = false),
    KDense(10, 10, 10; use_base_act = false),
    KDense(10,  1, 10; use_base_act = false),
) # 1_200 parameters, plus 30 states
p, st = Lux.setup(rng, kan) |> device
f(p) = mlp(x, p, st)[1] |> sum

@btime CUDA.@sync $kan($x, $p, $st) # 83.275 μs (310 allocations: 10.00 KiB)
@btime CUDA.@sync Zygote.gradient($f, $p) # 874.364 μs (2746 allocations: 99.70 KiB)
```
Although KANs are currently 2-3x slower than an MLPs with the same number of parameters,
the promise with this architecture is that a small KAN can potentially do the work of a much bigger MLP.
More experiments need to be done to assess the validity of this claim.

This package will be actively developed for the time-being.
Once a stable version is figured out, we can consider opening a PR on [`Lux.jl`](https://github.com/LuxDL/Lux.jl).
Feel fre to open issues or create PRs in the meantime with features, comparisons, or performance improvements.

# TODO:
- Grid update with linear least sq solve
- devise good initialization schemes. RBF coefficients and base activation weights are currently initialized with `WeightInitializers.glorot_uniform`.
- figure out what are good optimization strategies (choice of optimizer, learning rate decay, etc)
