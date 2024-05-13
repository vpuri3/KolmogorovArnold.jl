#
@inline function rbf(x, z, h) # exp(-((x - z)/h)^2)
    y = (x - z) * (1/h)
    gaussian1D(y)
end

@inline gaussian1D(x) = exp(-x^2)

function CRC.rrule(::typeof(gaussian1D), x)
    T = eltype(x)
    y = gaussian1D(x)
    ∇gaussian1D(ȳ) = -T(2) * x * y * ȳ

    return y, ∇gaussian1D
end

# from https://github.com/LuxDL/Lux.jl/pull/627
@inline silu(x) = x * sigmoid(x)
@inline silu_fast(x) = x * sigmoid_fast(x)
@inline NNlib.fast_act(::typeof(silu), ::AbstractArray=1:0) = silu_fast

