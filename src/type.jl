#
@concrete struct KDense <: LuxCore.AbstractExplicitLayer
    in_dims::Int
    out_dims::Int
    grid_len::Int
    denominator
    normalizer
    base_act
    init_W1
    init_W2
    use_base_activation
end

function KDense(
    in_dims::Int,
    out_dims::Int,
    grid_len::Int;
    denominator = Float32(2 / (grid_len - 1)),
    base_act = silu,
    init_W1 = glorot_uniform,
    init_W2 = glorot_uniform,
    use_base_activation = true,
    use_fast_activation::Bool = true,
)
    normalizer = use_fast_activation ? tanh_fast : tanh

    KDense(
        in_dims, out_dims, grid_len,
        denominator, normalizer, base_act,
        init_W1, init_W2, use_base_activation,
    )
end

function LuxCore.initialparameters(rng::AbstractRNG, l::KDense)
    p = (;
        W1 = l.init_W1(rng, l.out_dims, l.grid_len * l.in_dims),
    )

    if l.use_base_activation
        p = (;
            p..., W2 = l.init_W2(rng, l.out_dims, l.in_dims),
        )
    end

    p
end

function LuxCore.initialstates(::AbstractRNG, l::KDense,)
    grid = collect(LinRange(-1, 1, l.grid_len)) .|> Float32

    (; grid,)
end

LuxCore.statelength(l::KDense) = l.grid_len
function LuxCore.parameterlength(l::KDense)
    len = l.in_dims * l.grid_len * l.out_dims
    if l.use_base_activation
        len += l.in_dims * l.out_dims
    end

    len
end

function (l::KDense)(x::AbstractVecOrMat, p, st)
    K = size(x, 2)                # [I, K]
    x_resh = reshape(x, 1, :)     # [1, I * K]
    x_norm = l.normalizer(x_resh) # ∈ [-1, 1]

    basis = rbf.(x_norm, st.grid, l.denominator)       # [G, I * K]
    basis = reshape(basis, l.grid_len * l.in_dims, K) # [G * I, K]
    spline = p.W1 * basis                             # [O, K]

    y = if l.use_base_activation
        spline + p.W2 * l.base_act.(x)
    else
        spline
    end

    y, st
end 
#
