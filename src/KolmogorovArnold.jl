module KolmogorovArnold

using Random
using LinearAlgebra

using NNlib
using LuxCore
using WeightInitializers
using ConcreteStructs

using CUDA, cuTENSOR
using TensorOperations

using ChainRulesCore
const CRC = ChainRulesCore

include("utils.jl")
export rbf, rswaf, iqf, batched_mul

include("kdense.jl")
export KDense

include("fdense.jl")
export FDense

include("cdense.jl")
export CDense

# include("explicit")
# export GDense

end # module
