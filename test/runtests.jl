using KolmogorovArnold
using Test
using Lux, Zygote, Random, Statistics, Plots, CUDA, LuxCUDA, ComponentArrays
using Optimisers, OptimizationOptimJL, LuxDeviceUtils

pkgpath = dirname(dirname(pathof(KolmogorovArnold)))

@testset "FunctionFit" begin

    cpud = cpu_device()
    gpud = gpu_device()
    rng  = Random.default_rng()

    """
    Fits the model to the curve and returns the 
    """
    function fit(name, model, device, i_shape, i_d)
        
        x₀ = sort!(rand(Float32, i_shape...), dims=i_d) 
        yₜ = cos.(π .* x₀) .+ sin.(4 .* π .* x₀) .* tanh.(x₀) .+ x₀.^4 
        x₀ = x₀ |> device
        yₜ = yₜ |> device

        # Initiate model
        parameters, layer_states = Lux.setup(rng, model)
        parameters = ComponentArray(parameters) |> device
        layer_states = layer_states |> device

        # Initial Prediction
        yᵢ, layer_states = model(x₀, parameters, layer_states)

        # Set up the optimizer
        opt_state = Optimisers.setup(Optimisers.Adam(0.0003), parameters)

        # Define the loss function
        function loss_fn(pa, ls)
            yₚ, new_ls = model(x₀, pa, ls)
            l = mean((yₚ .- yₜ).^2)
            return l, new_ls
        end

        loss = 10.0f32
        epoch = 0
        while loss > 1e-4 && epoch < 2e4
            (loss, layer_states), back = pullback(loss_fn, parameters, layer_states)
            grad, _ = back((1.0, nothing))
            grad = map(g -> clamp(g, -3, 3), grad)
            opt_state, parameters = Optimisers.update(opt_state, parameters, grad)
            #print("\rName: $name - Epoch: $epoch, Loss: $loss")
            epoch += 1
        end

        #println()
        ## Getting the final evaluatin for the test
        #yₑ, layer_states = model(x₀, parameters, layer_states)
        ## Plotting the truth and the initial / final predictions
        #x₀ = vec(x₀) |> cpud
        #yᵢ = vec(yᵢ) |> cpud
        #yₑ = vec(yₑ) |> cpud
        #yₜ = vec(yₜ) |> cpud
        #plot(x₀, yₜ, label="truth", color="red")
        #plot!(x₀, yᵢ, label="inita approx", color="blue")
        #plot!(x₀, yₑ, label="final approx", color="green")
        #scatter!(x₀, yₜ, color="red", label=false)
        #scatter!(x₀, yᵢ, color="blue", label=false)
        #scatter!(x₀, yₑ, color="green", label=false)
        #savefig("$name.png")

        epoch
    end

    @test fit("fKAN_cpu", Chain(FDense(1, 10, 10), FDense(10, 1, 10)), cpud, (50, 1), 1) <= 2e4
    @test fit("cKAN_cpu", Chain(CDense(1, 20, 50), CDense(20, 1, 50)), cpud, (50, 1), 1) <= 2e4
    @test fit("rKAN_cpu", Chain(KDense(1, 10, 10), KDense(10, 1, 10)), cpud, (1, 50), 2) <= 2e4
    @test fit("fKAN_gpu", Chain(FDense(1, 10, 10), FDense(10, 1, 10)), gpud, (50, 1), 1) <= 2e4
    @test fit("cKAN_gpu", Chain(CDense(1, 20, 50), CDense(20, 1, 50)), gpud, (50, 1), 1) <= 2e4
    @test fit("rKAN_gpu", Chain(KDense(1, 10, 10), KDense(10, 1, 10)), gpud, (1, 50), 2) <= 2e4

end

@testset "Speedtest" begin
    include(joinpath(pkgpath, "examples", "eg1.jl"))
end

