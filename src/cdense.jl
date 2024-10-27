#
#======================================================================#
# Kolmogorov-Arnold Layer with chebyshev polynomials as basis functions
#======================================================================#
struct CDense <: LuxCore.AbstractExplicitLayer
    inputdim::Int
    outdim::Int
    degree::Int
    #
    init::Function
end

# Constructor
function CDense(
    inputdim::Int, 
    outdim::Int, 
    degree::Int,
    init::Function = glorot_uniform)

    CDense(inputdim, outdim, degree, init)
end


# Initialize parameters for the layer
function LuxCore.initialparameters(rng::AbstractRNG, l::CDense)
    p = (; C = l.init(rng, Float32, l.inputdim, l.outdim, l.degree + 1) .* (1 / (l.inputdim * (l.degree + 1))))
    p = (p..., W = collect(Float32, 0:l.degree))
    p
end


# Forward pass
function (l::CDense)(x::AbstractArray, p, st)

    x = tanh.(x)                        # Apply tanh to normalize x to [-1, 1]

    x = reshape(x, :, l.inputdim, 1)    # Shape: (batch_size, inputdim, 1)
    x = repeat(x, 1, 1, l.degree + 1)   # Expand: (batch_size, inputdim, degree + 1)
    
    x = acos.(x)                        # Apply acos
    x .= x .* p.W                       # Multiply by arange [0 .. degree]
    x = cos.(x)                         # Apply cos

    # Compute Chebyshev interpolation using einsum equivalent (batched multiplication)
    y = batched_mul(x, p.C)  # Equivalent to einsum "bid,iod->bo"

    # Flatten the result to have shape (batch_size, outdim)
    return reshape(y, :, l.outdim), st
end
