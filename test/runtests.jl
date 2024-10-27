using KolmogorovArnold
using Test
using Lux, Zygote, Random, Statistics, Plots
using Optimisers, OptimizationOptimJL

# Write your tests here.


@testset "FunctionFit.jl" begin

    rng = Random.default_rng()

    x₀ = rand(Float32, 50, 1)
    sort!(x₀, dims=1)
    yₜ = cos.(π .* x₀) .+ sin.(4 .* π .* x₀) .* tanh.(x₀) .+ x₀ .^ 4


    model = Chain(CDense(1, 10, 10), CDense(10, 1, 10))
    parameters, layer_states = Lux.setup(rng, model)

    y_init, layer_states = model(x₀, parameters, layer_states)

    # Define the loss function
    function loss_fn(pa, ls)
        yₚ, new_ls = model(x₀, pa, ls)
        l = mean((yₚ .- yₜ).^2)
        return l, new_ls
    end

    # Set up the optimizer
    opt = Descent(0.01)
    opt_state = Optimisers.setup(opt, parameters)

    for epoch in 1:10000
        (loss, layer_states), back = pullback(loss_fn, parameters, layer_states)
        grad, _ = back((1.0, nothing))
        opt_state, parameters = Optimisers.update(opt_state, parameters, grad)
        if epoch % 1000 == 0
            println("Epoch: $epoch, Loss: $loss")
        end
    end

    y_last, layer_states = model(x₀, parameters, layer_states)


    p = plot(x₀, yₜ, label="truth", color="red")
    plot!(x₀, y_init, label="inita approx", color="blue")
    plot!(x₀, y_last, label="final approx", color="green")
    scatter!(x₀, yₜ, color="red")
    scatter!(x₀, y_init, color="blue")
    scatter!(x₀, y_last, color="green")
    savefig(p, "p.png")
end
