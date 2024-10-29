#
#======================================================#
# Kolmogorov-Arnold Layer with fourier basis functions
#======================================================#
@concrete struct FDense{addbias} <: LuxCore.AbstractExplicitLayer
    inputdim::Int
    outdim::Int
    gridsize::Int
    #
    init
    #
    grid_norm_factor
end


function FDense(
    inputdim::Int,
    outdim::Int,
    gridsize::Int;
    addbias::Bool = true,
    smooth_initialization::Bool = false,
    init = glorot_uniform
)

    @assert gridsize > 0


    # Grid normalization factor
    grid_norm_factor = smooth_initialization ? ((1:gridsize).^2) : sqrt(gridsize)
    
    FDense{addbias}(inputdim, outdim, gridsize, init, grid_norm_factor)
end

# Initialize parameters for the layer
function LuxCore.initialparameters(rng::AbstractRNG, l::FDense{addbias}) where {addbias}
    p = (;fouriercoeffs = l.init(rng, Float32, 2, l.outdim, l.inputdim, l.gridsize) ./ (sqrt(l.inputdim) * l.grid_norm_factor))

    if addbias
        p = (; p..., bias = zeros(Float32, 1, l.outdim))
    end

    p
end

# Compute the number of parameters
function LuxCore.parameterlength(l::FDense{addbias}) where {addbias}
    length = 2 * l.outdim * l.inputdim * l.gridsize

    if addbias
        length += l.outdim
    end

    length
end


# Forward pass implementation
function (l::FDense{addbias})(x::AbstractArray, p, st) where {addbias}

    size_in  = size(x)
    outshape = (size_in[1:end-1]..., l.outdim)

    x     = reshape(x, :, l.inputdim)
    k     = reshape(1:l.gridsize, 1, 1, 1, l.gridsize)
    xrshp = reshape(x, size_in[1], 1, size_in[2], 1)

    # Compute cos(k*x) and sin(k*x)
    c = cos.(k .* xrshp)
    s = sin.(k .* xrshp)

    # Compute the Fourier interpolations for each input dimension
    y = sum(c .* p.fouriercoeffs[1:1, :, :, :], dims = (3, 4)) + sum(s .* p.fouriercoeffs[2:2, :, :, :], dims = (3, 4))
    y = dropdims(y, dims=(3, 4)) 

    # Add bias if needed
    if addbias
        y = y .+ p.bias
    end

    # Reshape output to expected output shape
    return reshape(y, outshape), st
end
