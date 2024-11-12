#
using KolmogorovArnold

# Add test dependencies to env stack
let 
    pkgpath = dirname(dirname(pathof(KolmogorovArnold)))
    tstpath = joinpath(pkgpath, "test")
    !(tstpath in LOAD_PATH) && push!(LOAD_PATH, tstpath)
end

using Random, LinearAlgebra
using MLDataDevices, BenchmarkTools
using Enzyme, Zygote, Lux, ComponentArrays

# configure BLAS
ncores = min(Sys.CPU_THREADS, length(Sys.cpu_info()))
BLAS.set_num_threads(ncores)

# configure CUDA
using CUDA, LuxCUDA
CUDA.allowscalar(false)

# configure Reactant
using Reactant
Reactant.set_default_backend("gpu")

rng = Random.default_rng()
Random.seed!(rng, 0)

device_zy = gpu_device()
device_ra = reactant_device()

function Reactant.synchronize(x::ComponentArray)
	Reactant.synchronize(getdata(x))
	ComponentArray(getdata(x), getaxes(x))
end

#======================================================#
function main()

    x = rand32(rng, 1, 100)
	y = x .^ 2

    wM, wK, G = 128, 40, 10 # MLP width, KAN width, grid size

    mlp = Chain(
        Dense(1, wM, tanh),
        Dense(wM, wM, tanh),
        Dense(wM, 1),
    )

    basis_func = rbf      # rbf, rswaf
    normalizer = softsign # sigmoid(_fast), tanh(_fast), softsign
    
    kan1 = Chain(
        KDense( 1, wK, G; use_base_act = true, basis_func, normalizer),
        KDense(wK, wK, G; use_base_act = true, basis_func, normalizer),
        KDense(wK,  1, G; use_base_act = true, basis_func, normalizer),
    )

    kan2 = Chain(
        KDense( 1, wK, G; use_base_act = false, basis_func, normalizer),
        KDense(wK, wK, G; use_base_act = false, basis_func, normalizer),
        KDense(wK,  1, G; use_base_act = false, basis_func, normalizer),
    )

    # display(mlp)
    # display(kan1)
    # display(kan2)

    pM, stM = Lux.setup(rng, mlp)
    pK1, stK1 = Lux.setup(rng, kan1)
    pK2, stK2 = Lux.setup(rng, kan2)

    pM = ComponentArray(pM)
    pK1 = ComponentArray(pK1)
    pK2 = ComponentArray(pK2)

	function loss(model, ps, st, x, y)
		pred, _ = model(x, ps, st)
		return MSELoss()(pred, y)
	end

	#------------------------#
	# set up Reactant / Zygote
	#------------------------#
	x_zy = x |> device_zy
	y_zy = y |> device_zy

	pM_zy , stM_zy  = (pM , stM ) .|> device_zy
	pK1_zy, stK1_zy = (pK1, stK1) .|> device_zy
	pK2_zy, stK2_zy = (pK2, stK2) .|> device_zy

	function grad_zy(model, ps, st, x, y)
		lossfun = ps -> loss(model, ps, st, x, y)
		only(Zygote.gradient(lossfun, ps))
	end

	#------------------------#
	x_ra = x |> device_ra
	y_ra = y |> device_ra

	pM_ra , stM_ra  = (pM , stM ) .|> device_ra
	pK1_ra, stK1_ra = (pK1, stK1) .|> device_ra
	pK2_ra, stK2_ra = (pK2, stK2) .|> device_ra

	function grad_ra(model, ps, st, x, y)
		Enzyme.gradient(Enzyme.Reverse, Const(loss), Const(model),
			ps, Const(st), Const(x), Const(y))[2]
	end

	mlp_comp  = @compile mlp( x_ra, pM_ra , stM_ra )
	kan1_comp = @compile kan1(x_ra, pK1_ra, stK1_ra)
	kan2_comp = @compile kan2(x_ra, pK2_ra, stK2_ra)

	grad_ra_comp_M  = @compile grad_ra(mlp , pM_ra, stM_ra, x_ra, y_ra)
	grad_ra_comp_K1 = @compile grad_ra(kan1, pK1_ra, stK1_ra, x_ra, y_ra)
	grad_ra_comp_K2 = @compile grad_ra(kan2, pK2_ra, stK2_ra, x_ra, y_ra)

	#------------------------#

	println("\n# FWD Vanilla\n")
	
	@btime CUDA.@sync $mlp( $x_zy, $pM_zy , $stM_zy )
	@btime CUDA.@sync $kan1($x_zy, $pK1_zy, $stK1_zy)
	@btime CUDA.@sync $kan2($x_zy, $pK2_zy, $stK2_zy)
	
	println("\n# FWD Reactant\n")
	
	@btime Reactant.synchronize($mlp_comp( $x_ra, $pM_ra , $stM_ra )[1])
	@btime Reactant.synchronize($kan1_comp($x_ra, $pK1_ra, $stK1_ra)[1])
	@btime Reactant.synchronize($kan2_comp($x_ra, $pK2_ra, $stK2_ra)[1])

	#------------------------#
	println("\n# BWD Zygote\n")
	
	@btime CUDA.@sync $grad_zy($mlp , $pM , $stM , $x, $y)
	@btime CUDA.@sync $grad_zy($kan1, $pK1, $stK1, $x, $y)
	@btime CUDA.@sync $grad_zy($kan2, $pK2, $stK2, $x, $y)

	println("\n# BWD Reactant\n")

	@btime Reactant.synchronize($grad_ra_comp_M( $mlp , $pM_ra , $stM_ra , $x_ra, $y_ra))
	@btime Reactant.synchronize($grad_ra_comp_K1($kan1, $pK1_ra, $stK1_ra, $x_ra, $y_ra))
	@btime Reactant.synchronize($grad_ra_comp_K2($kan2, $pK2_ra, $stK2_ra, $x_ra, $y_ra))
	#------------------------#

	return
end

#======================================================#
main()
