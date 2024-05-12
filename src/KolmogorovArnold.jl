module KolmogorovArnold

using Random
using LinearAlgebra

using NNLib
using LuxCore
using WeightInitializers
using ConcreteStructs

include("utils.jl")

include("type.jl")
export KDense

end # module
