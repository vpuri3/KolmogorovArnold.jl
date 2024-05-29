module KolmogorovArnold

using Random
using LinearAlgebra

using NNlib
using LuxCore
using WeightInitializers
using ConcreteStructs

using ChainRulesCore
const CRC = ChainRulesCore

include("utils.jl")
export rbf, rswaf, iqf

include("kdense.jl")
export KDense

# include("explicit")
# export GDense

end # module
