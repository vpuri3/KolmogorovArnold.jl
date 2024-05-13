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

function (l::KDense{false})(x::AbstractVecOrMat, p, st)
    K = size(x, 2)                # [I, K]
    x_resh = reshape(x, 1, :)     # [1, I * K]
    x_norm = l.normalizer(x_resh) # ∈ [-1, 1]

    basis = rbf.(x_norm, st.grid, l.denominator)      # [G, I * K]
    basis = reshape(basis, l.grid_len * l.in_dims, K) # [G * I, K]
    y = p.W1 * basis                                  # [O, K]

    y, st
end 

function (l::KDense{true})(x::AbstractVecOrMat, p, st)
    K = size(x, 2)                # [I, K]
    x_resh = reshape(x, 1, :)     # [1, I * K]
    x_norm = l.normalizer(x_resh) # ∈ [-1, 1]

    basis = rbf.(x_norm, st.grid, l.denominator)      # [G, I * K]
    basis = reshape(basis, l.grid_len * l.in_dims, K) # [G * I, K]
    spline = p.W1 * basis                             # [O, K]

    y = spline + p.W2 * l.base_act.(x)

    y, st
end 
#
