#
@inline function rbf(x, z, h)
    # exp(-((x - z)/h)^2)
    y = (x - z) / h
    negative_exp_square(y)
end

@inline negative_exp_square(x) = exp(-x^2)

function CRC.rrule(::typeof(negative_exp_square), x)
    T = eltype(x)
    y = negative_exp_square(x)
    ∇negative_exp_square(ȳ) = -T(2) * x * y * ȳ

    return y, ∇negative_exp_square
end

# from https://github.com/LuxDL/Lux.jl/pull/627
@inline silu(x) = x * sigmoid(x)
@inline silu_fast(x) = x * sigmoid_fast(x)
@inline NNlib.fast_act(::typeof(silu), ::AbstractArray=1:0) = silu_fast

