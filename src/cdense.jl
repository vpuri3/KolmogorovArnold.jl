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
    addbias = true)

    CDense{addbias}(inputdim, outdim, degree, init)
end


# Initialize parameters for the layer
function LuxCore.initialparameters(rng::AbstractRNG, l::CDense{addbias}) where {addbias}
    p = (;chebycoeffs = l.init(rng, Float32, l.inputdim, l.outdim, l.degree + 1) .* (1 / (l.inputdim * (l.degree + 1))))
    p = (;p..., arange = collect(Float32, 0:l.degree))

    p = if addbias
        (;p..., B = zeros(Float32, 1, l.outdim))
    end

    p
end

# Compute the number of parameters
function LuxCore.parameterlength(l::CDense{addbias}) where {addbias}
    length = l.outdim * l.inputdim * (l.degree + 1)
    
    if addbias
        length += l.outdim
    end

    length
end


# Forward pass
function (l::CDense{addbias})(x::AbstractArray, p, st) where {addbias}

    x = tanh.(x)
    x = reshape(x, :, l.inputdim, 1)
    x = repeat(x, 1, 1, l.degree + 1)

    x = acos.(x)
    x = x .+ reshape(p.arange, 1, 1, :)
    x = cos.(x)

    y = batched_mul(x, p.chebycoeffs)  # Equivalent to einsum "bid,iod->bo"

    if addbias
        y = y .+ p.B
    end

    reshape(y, :, l.outdim), st
end
