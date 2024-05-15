#
#======================================================#
# Kolmogorov-Arnold Layer
#======================================================#
@concrete struct KDense{use_base_act} <: LuxCore.AbstractExplicitLayer
    in_dims::Int
    out_dims::Int
    grid_len::Int
    #
    normalizer
    grid_lims
    denominator
    #
    base_act
    init_C
    init_W
end

function KDense(
    in_dims::Int,
    out_dims::Int,
    grid_len::Int;
    #
    normalizer = tanh,
    grid_lims::NTuple{2, Real} = (-1.0f0, 1.0f0),
    denominator = Float32(2 / (grid_len - 1)),
    #
    base_act = swish,
    use_base_act = true,
    #
    init_C = glorot_uniform,
    init_W = glorot_uniform,
    use_fast_act::Bool = true,
)
    T = promote_type(eltype.(grid_lims)...)

    if isnothing(grid_lims)
        grid_lims = if normalizer ∈ (sigmoid, sigmoid_fast)
            (0, 1)
        elseif normalizer ∈ (tanh, tanh_fast, softsign)
            (-1, 1)
        else
            (-1, 1)
        end
    end

    grid_span =  grid_lims[2] > grid_lims[1]
    @assert grid_span > 0

    if isnothing(denominator)
        denominator = grid_span / (grid_len - 1)
    end

    if use_fast_act
        base_act = NNlib.fast_act(base_act)
        normalizer = NNlib.fast_act(normalizer)
    end

    KDense{use_base_act}(
        in_dims, out_dims, grid_len,
        normalizer, T.(grid_lims), T(denominator),
        base_act, init_C, init_W,
    )
end

function LuxCore.initialparameters(
    rng::AbstractRNG,
    l::KDense{use_base_act}
) where{use_base_act}
    p = (;
        C = l.init_C(rng, l.out_dims, l.grid_len * l.in_dims),
    )
    # C = l.init_C(rng, l.out_dims, l.grid_len,  l.in_dims),

    if use_base_act
        p = (;
            p...,
            W = l.init_W(rng, l.out_dims, l.in_dims),
        )
    end

    p
end

function LuxCore.initialstates(::AbstractRNG, l::KDense,)
    (;
        grid = collect(LinRange(l.grid_lims..., l.grid_len))
    )
end

function LuxCore.statelength(l::KDense)
    l.grid_len
end

function LuxCore.parameterlength(
    l::KDense{use_base_act},
) where{use_base_act}
    len = l.in_dims * l.grid_len * l.out_dims
    if use_base_act
        len += l.in_dims * l.out_dims
    end

    len
end

function (l::KDense{use_base_act})(x::AbstractArray, p, st) where{use_base_act}
    size_in  = size(x)                          # [I, ..., batch,]
    size_out = (l.out_dims, size_in[2:end]...,) # [O, ..., batch,]

    x = reshape(x, l.in_dims, :)
    K = size(x, 2)

    x_norm = l.normalizer.(x)                          # ∈ [-1, 1]
    x_resh = reshape(x_norm, 1, :)                     # [1, K]
    basis  = rbf.(x_resh, st.grid, l.denominator)      # [G, I * K]
    basis  = reshape(basis, l.grid_len * l.in_dims, K) # [G * I, K]
    spline = p.C * basis                               # [O, K]

    y = if use_base_act
        base = p.W * l.base_act.(x)
        spline + base
    else
        spline
    end

    reshape(y, size_out), st
end
#======================================================#
#
