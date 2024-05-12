# KolmogorovArnold.jl

[![Build Status](https://github.com/vpuri3/KolmogorovArnold.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/vpuri3/KolmogorovArnold.jl/actions/workflows/CI.yml?query=branch%3Amaster)

Julia implementation of the [Kolmogorov-Arnold network](https://arxiv.org/abs/2404.19756)
for the [`Lux.jl`](https://lux.csail.mit.edu/stable/) framework.
This implementation is based on [`efficient-kan`](https://github.com/Blealtan/efficient-kan)
and ['FastKAN'](https://github.com/ZiyaoLi/fast-kan) which resolve the performance
issues with the [original implementation](https://github.com/KindXiaoming/pykan).

```julia
using Lux, KolmogorovArnold

x = rand32()
```
