#
#======================================================================#
# Kolmogorov-Arnold Layer with chebyshev polynomials as basis functions
#======================================================================#
@concrete struct CDense{addbias}  <: LuxCore.AbstractExplicitLayer
    inputdim::Int
    outdim::Int
    degree::Int
    #
    init
end

# Constructor
function CDense(
    inputdim::Int, 
    outdim::Int, 
    degree::Int,
    init::Function = glorot_uniform,
    add_bias = true)

    CDense{add_bias}(inputdim, outdim, degree, init)
end


# Initialize parameters for the layer
function LuxCore.initialparameters(rng::AbstractRNG, l::CDense{add_bias}) where {add_bias}
    C = l.init(rng, Float32, l.inputdim, l.outdim, l.degree + 1) .* (1 / (l.inputdim * (l.degree + 1)))
    W = collect(Float32, 0:l.degree)
    p = (C = C, W = W)
    if add_bias
        B = zeros(Float32, 1)
        p = (p..., B = B)
    end

    return p
end


# Forward pass
function (l::CDense{add_bias})(x::AbstractArray, p, st) where {add_bias}

    x = tanh.(x)
    x = reshape(x, :, l.inputdim, 1)
    x = repeat(x, 1, 1, l.degree + 1)

    x = acos.(x)
    x = x .+ reshape(p.W, 1, 1, :)
    x = cos.(x)

    y = batched_mul(x, p.C)  # Equivalent to einsum "bid,iod->bo"

    if add_bias
        y = y .+ p.B
    end

    return reshape(y, :, l.outdim), st
end
