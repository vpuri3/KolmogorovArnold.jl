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

Compared to an MLP with the same number of parameters, a Kolmogorov-Arnold network (KAN)
consumes 6x more memory and is 4x slower.
```julia
using Lux, KolmogorovArnold

mlp = Chain(Dense(1, 32, tanh), Dense(32, 32, tanh), Dense(32, 1),)
```
```julia
Chain(
    layer_1 = Dense(1 => 32, tanh_fast),  # 64 parameters
    layer_2 = Dense(32 => 32, tanh_fast),  # 1_056 parameters
    layer_3 = Dense(32 => 1),           # 33 parameters
)         # Total: 1_153 parameters,
          #        plus 0 states.
```

```julia
kan = Chain(KDense(1, 8, 15), KDense(8, 8, 15), KDense(8, 1, 15))
```
```julia
Chain(
    layer_1 = KDense(),                 # 128 parameters, plus 15
    layer_2 = KDense(),                 # 1_024 parameters, plus 15
    layer_3 = KDense(),                 # 128 parameters, plus 15
)         # Total: 1_280 parameters,
          #        plus 45 states.
```
```julia
using CUDA, LuxCUDA
device = Lux.gpu_device()

x = rand32(rng, 1, 1000) |> device
pM, stM = Lux.setup(rng, mlp) |> device
pK, stK = Lux.setup(rng, kan) |> device

@btime CUDA.@sync $mlp($x, $pM, $stM)
@btime CUDA.@sync $kan($x, $pK, $stK)
```
```julia
  34.360 μs (175 allocations: 4.78 KiB)
  155.781 μs (565 allocations: 17.50 KiB)
```
With `use_base_activcation = false`, the performance of KAN effectively doubles
```julia
kan = Chain(
    KDense(1, 8, 15; use_base_activation = false),
    KDense(8, 8, 15; use_base_activation = false),
    KDense(8, 1, 15; use_base_activation = false),
)
p, st = Lux.setup(rng, kan) |> device
@btime CUDA.@sync $mlp($x, $p, $st)
```
```julia
  83.275 μs (310 allocations: 10.00 KiB)
```
The promise with this architecture is that a small KAN can potentially do the work of a
much bigger MLP.
More experiments need to be done to assess the validity of this claim.

This package will be actively developed for the time-being.
Once a stable version is figured out, we can consider opening a PR on [`Lux.jl`](https://github.com/LuxDL/Lux.jl).
Feel fre to open issues or create PRs in the meantime with features, comparisons, or performance improvements.
