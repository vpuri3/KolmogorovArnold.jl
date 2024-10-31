# KolmogorovArnold.jl

[![Build Status](https://github.com/vpuri3/KolmogorovArnold.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/vpuri3/KolmogorovArnold.jl/actions/workflows/CI.yml?query=branch%3Amaster)

Julia implementation of [FourierKAN](https://github.com/GistNoesis/FourierKAN)

Julia implementation of [ChebyKAN](https://github.com/SynodicMonth/ChebyKAN)


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

We compare the performance of different implementation of KAN with an MLP that has the same number of parameters (see `examples/eg1.jl`).
```julia
using Lux, KolmogorovArnold
using CUDA, LuxDeviceUtils

CUDA.allowscalar(false)
device = Lux.gpu_device()

rng = Random.default_rng()
Random.seed!(rng, 0)

x  = rand32(rng, 1, 1000) |> device
x₀ = rand32(rng, 1000, 1) |> device
# define MLP, KANs

mlp = Chain(
    Dense(1, 128, tanh),
    Dense(128, 128, tanh),
    Dense(128, 1),
) # 16_897 parameters plus 0 states.

basis_func = rbf # rbf, rswaf, iqf (radial basis funcs, reflection switch activation funcs, inverse quadratic funcs)
normalizer = softsign # sigmoid(_fast), tanh(_fast), softsign

kan1 = Chain(
    KDense( 1, 40, 10; use_base_act = true, basis_func, normalizer),
    KDense(40, 40, 10; use_base_act = true, basis_func, normalizer),
    KDense(40,  1, 10; use_base_act = true, basis_func, normalizer),
) # 18_490 parameters plus 30 states.

kan2 = Chain(
    KDense( 1, 40, 10; use_base_act = false, basis_func, normalizer),
    KDense(40, 40, 10; use_base_act = false, basis_func, normalizer),
    KDense(40,  1, 10; use_base_act = false, basis_func, normalizer),
) # 16_800 parameters plus 30 states.

kan3 = Chain(
    CDense( 1, 40, G),
    CDense(40, 40, G),
    CDense(40,  1, G),
) # 18_561 parameters plus 0 states.

kan4 = Chain(
    FDense( 1, 30, G),
    FDense(30, 30, G),
    FDense(30,  1, G),
) # 19_261 parameters plus 0 states.

# set up experiment
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

# Forward pass

@btime CUDA.@sync $mlp($x, $pM, $stM)     # 31.611 μs (248 allocations: 5.45 KiB)
@btime CUDA.@sync $kan1($x, $pK1, $stK1)  # 125.790 μs (1034 allocations: 21.97 KiB)
@btime CUDA.@sync $kan2($x, $pK2, $stK2)  # 87.585 μs (1335 allocations: 13.95 KiB)
@btime CUDA.@sync $kan3($x', $pK3, $stK3) # 210.785 μs (1335 allocations: 31.03 KiB)
@btime CUDA.@sync $kan4($x', $pK4, $stK4) # 2.392 ms (1642 allocations: 34.56 KiB)


# Backward pass

f_mlp(p)  = mlp(x, p, stM)[1] |> sum
f_kan1(p) = kan1(x, p, stK1)[1] |> sum
f_kan2(p) = kan2(x, p, stK2)[1] |> sum
f_kan3(p) = kan3(x₀, p, stK3)[1] |> sum
f_kan4(p) = kan4(x₀, p, stK4)[1] |> sum


@btime CUDA.@sync Zygote.gradient($f_mlp, $pM)   # 268.074 μs (1971 allocations: 57.03 KiB)
@btime CUDA.@sync Zygote.gradient($f_kan1, $pK1) # 831.888 μs (5015 allocations: 123.25 KiB)
@btime CUDA.@sync Zygote.gradient($f_kan2, $pK2) # 658.578 μs (3314 allocations: 87.16 KiB)
@btime CUDA.@sync Zygote.gradient($f_kan3, $pK3) # 1.647 ms (7138 allocations: 180.45 KiB)
@btime CUDA.@sync Zygote.gradient($f_kan4, $pK4) # 7.028 ms (8745 allocations: 199.42 KiB)


```
The performance of KAN with radial basis functions improves significantly with `use_base_act = false`.
Although KANs are currently significantly slower than an MLPs with the same number of parameters,
the promise with this architecture is that a small KAN can potentially do the work of a much bigger MLP.
More experiments need to be done to assess the validity of this claim.

This package will be actively developed for the time-being.
Once a stable version is figured out, we can consider opening a PR on [`Lux.jl`](https://github.com/LuxDL/Lux.jl).
Feel fre to open issues or create PRs in the meantime with features, comparisons, or performance improvements.

## Custom gradients

Writing custom gradients for the activation function has led to substantial speedup in the backward pass (see `examples/eg2.jl`).
```julia
N, G = 5000, 10

x = LinRange(-1, 1, N) |> Array |> device
z = LinRange(-1, 1, G) |> Array |> device
d = 2 / (G-1)

f_rbf(z)   = rbf(  x, z', d) |> sum
f_rswaf(z) = rswaf(x, z', d) |> sum
f_iqf(z)   = iqf(  x, z', d) |> sum

# forward pass
@btime CUDA.@sync $f_rbf($z)   # 55.566 μs (294 allocations: 7.78 KiB)
@btime CUDA.@sync $f_rswaf($z) # 57.112 μs (294 allocations: 7.78 KiB)
@btime CUDA.@sync $f_iqf($z)   # 55.368 μs (294 allocations: 7.78 KiB)

# backward pass
@btime CUDA.@sync Zygote.gradient($f_rbf  , $z) # 188.456 μs (1045 allocations: 27.62 KiB)
@btime CUDA.@sync Zygote.gradient($f_rswaf, $z) # 212.419 μs (1071 allocations: 28.30 KiB)
@btime CUDA.@sync Zygote.gradient($f_iqf  , $z) # 201.568 μs (1045 allocations: 27.62 KiB)

# without custom gradients
# RBF  : 250.313 μs (1393 allocations: 35.95 KiB)
# RSWAF: 282.864 μs (1389 allocations: 36.62 KiB)
# IQF  : 333.843 μs (1628 allocations: 42.70 KiB)
```

## TODO
- Grid update with linear least sq solve
- devise good initialization schemes. RBF coefficients and base activation weights are currently initialized with `WeightInitializers.glorot_uniform`.
- figure out what are good optimization strategies (choice of optimizer, learning rate decay, etc)
