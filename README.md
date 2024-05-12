# KolmogorovArnold.jl

[![Build Status](https://github.com/vpuri3/KolmogorovArnold.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/vpuri3/KolmogorovArnold.jl/actions/workflows/CI.yml?query=branch%3Amaster)

Julia implementation of the [Kolmogorov-Arnold network](https://arxiv.org/abs/2404.19756)
for the [`Lux.jl`](https://lux.csail.mit.edu/stable/) framework.
This implementation is based on [`efficient-kan`](https://github.com/Blealtan/efficient-kan)
and ['FastKAN'](https://github.com/ZiyaoLi/fast-kan) which resolve the performance
issues with the [original implementation](https://github.com/KindXiaoming/pykan).

```julia
using Random, KolmogorovArnold

in_dim = 4
out_dim = 4
grid_len = 8

rng = Random.default_rng()
p, st = Lux.setup(rng, layer)

layer(rand32(rng, in_dim, 10), p, st)
```

Compared to an MLP with the same number of parameters, a Kolmogorov-Arnold network (KAN)
consumes 6x more memory and is 4x slower.
```julia
using Lux, KolmogorovArnold

mlp = Chain(Dense(1, 32), Dense(32, 32), Dense(32, 1),)
```
```julia
Chain(
    layer_1 = Dense(1 => 32),           # 64 parameters
    layer_2 = Dense(32 => 32),          # 1_056 parameters
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
  26.551 μs (137 allocations: 3.22 KiB)
  156.638 μs (565 allocations: 17.50 KiB)
```
The promise with this architecture, however, is that a smaller KAN can do the work of a
much bigger MLP.
More experiments need to be done to assess its performance in relation to MLPs.

This package will be actively developed over the next few weeks. Feel fre to open issues
or create PRs with performance improvements.
