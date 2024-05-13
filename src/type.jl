#
#======================================================#
# Kolmogorov-Arnold Layer
#======================================================#
@concrete struct KDense{use_base_act} <: LuxCore.AbstractExplicitLayer
    in_dims::Int
    out_dims::Int
    grid_len::Int
    denominator
    normalizer
    base_act
    init_W1
    init_W2
end

function KDense(
    in_dims::Int,
    out_dims::Int,
    grid_len::Int;
    denominator = Float32(2 / (grid_len - 1)),
    base_act = silu,
    init_W1 = glorot_uniform,
    init_W2 = glorot_uniform,
    use_base_act = true,
    use_fast_act::Bool = true,
)
    normalizer = use_fast_act ? tanh_fast : tanh
    base_act = use_fast_act ? NNlib.fast_act(base_act) : base_act

    KDense{use_base_act}(
        in_dims, out_dims, grid_len,
        denominator, normalizer, base_act,
        init_W1, init_W2,
    )
end

function LuxCore.initialparameters(
    rng::AbstractRNG,
    l::KDense{use_base_act}
) where{use_base_act}
    p = (;
        W1 = l.init_W1(rng, l.out_dims, l.grid_len * l.in_dims),
    )

    # W1 = reshape(W1, l.out_dims, l.in_dims, l.grid_len)

    if use_base_act
        p = (;
            p...,
            W2 = l.init_W2(rng, l.out_dims, l.in_dims),
        )
    end

    p
end

function LuxCore.initialstates(::AbstractRNG, l::KDense,)
    grid = collect(LinRange(-1, 1, l.grid_len)) .|> Float32

    (; grid,)
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

    x_norm = l.normalizer(x)                           # ∈ [-1, 1]
    x_resh = reshape(x_norm, 1, :)                     # [1, K]
    basis  = rbf.(x_resh, st.grid, l.denominator)      # [G, I * K]
    basis  = reshape(basis, l.grid_len * l.in_dims, K) # [G * I, K]
    spline = p.W1 * basis                              # [O, K]

    # would tullio speed this up?
    # @tullio spline[O, K] = W1[O, I, G] * basis[G, I, K]

    y = if use_base_act
        base = p.W2 * l.base_act.(x)
        spline + base
    else
        spline
    end

    reshape(y, size_out), st
end
#
