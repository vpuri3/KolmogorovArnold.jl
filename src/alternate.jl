#
#======================================================#
# An alternate KAN Dense layer.
# Had to implement this to confirm that it doesn't train well
#======================================================#
export KDense1
@concrete struct KDense1{use_base_act} <: LuxCore.AbstractExplicitLayer
    in_dims::Int
    out_dims::Int
    grid_len::Int
    denominator
    normalizer
    base_act
    init_C
    init_W
end

function KDense1(
    in_dims::Int,
    out_dims::Int,
    grid_len::Int;
    denominator = Float32(2 / (grid_len - 1)),
    base_act = silu,
    init_C = glorot_uniform,
    init_W = glorot_uniform,
    use_base_act = true,
    use_fast_act::Bool = true,
)
    normalizer = use_fast_act ? tanh_fast : tanh
    base_act = use_fast_act ? NNlib.fast_act(base_act) : base_act

    KDense1{use_base_act}(
        in_dims, out_dims, grid_len,
        denominator, normalizer, base_act,
        init_C, init_W,
    )
end

function LuxCore.initialparameters(rng::AbstractRNG, l::KDense1)
    (;
        C = l.init_C(rng, l.grid_len, l.in_dims),
        W = l.init_W(rng, l.out_dims, l.in_dims),
    )
end

function LuxCore.initialstates(::AbstractRNG, l::KDense1,)
    grid = collect(LinRange(-1, 1, l.grid_len)) .|> Float32

    (; grid,)
end

LuxCore.statelength(l::KDense1) = l.grid_len
LuxCore.parameterlength(l::KDense1) = l.in_dims * (l.grid_len + l.out_dims)

function (l::KDense1{use_base_act})(x::AbstractArray, p, st) where{use_base_act}
    size_in  = size(x)                          # [I, ..., batch,]
    size_out = (l.out_dims, size_in[2:end]...,) # [O, ..., batch,]

    x = reshape(x, l.in_dims, :)
    K = size(x, 2)

    x_norm = l.normalizer(x)                      # ∈ [-1, 1]
    x_resh = reshape(x_norm, 1, l.in_dims, K)     # [I, K]
    basis  = rbf.(x_resh, st.grid, l.denominator) # [G, I, K]
    spline = dropdims(sum(p.C .* basis; dims = 1); dims = 1) # [I, K]
    y = use_base_act ? (spline + l.base_act.(x)) : spline
    z = p.W * y # [O, K]

    reshape(z, size_out), st
end
#======================================================#
#
