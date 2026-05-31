
@testset "Bundles" begin
    logl(x::AbstractVector) =  exp(-x[1]^2 / 2) / √(2π)
    priors = [Uniform(-1, 1)]
    model = NestedModel(logl, priors)
    spl = Nested(1, 500)
    chains, _ = sample(rng, model, spl; dlogz=0.2, param_names=["x"], chain_type=Chains)
    val_arr, _ = sample(rng, model, spl; dlogz=0.2, chain_type=Array)

    @test size(chains, 2) == size(val_arr, 2)

    # test with add_live = false
    chains2, _ = sample(rng, model, spl; add_live=false, dlogz=0.2, param_names=["x"], chain_type=Chains)
    val_arr2, _ = sample(rng, model, spl; add_live=false, dlogz=0.2, chain_type=Array)
    
    @test size(chains2, 2) == size(val_arr2, 2)
    @test size(chains2, 1) < size(chains, 1) && size(val_arr2, 1) < size(val_arr, 1)

    # test check_wsum kwarg
    chains3, _ = sample(rng, model, spl; dlogz=0.2, param_names=["x"], chain_type=Chains)
    val_arr3, _ = sample(rng, model, spl; dlogz=0.2, chain_type=Array)

    @test size(chains3, 2) == size(val_arr3, 2)
end

@testset "Zero likelihood" begin
    logl(x::AbstractVector) = x[1] > 0 ? exp(-x[1]^2 / 2) / √(2π) : -Inf
    priors = [Uniform(-1, 1)]
    model = NestedModel(logl, priors)
    spl = Nested(1, 500)
    chains, _ = sample(rng, model, spl; param_names=["x"])
    @test all(>(0), chains[:x][chains[:weights] .> 1e-10])
end

@testset "logzerr stability with -Inf regions" begin
    # Regression: a likelihood with a forbidden (-Inf) region makes init_particles
    # clamp those live points to the -1e300 sentinel. The old information/logzerr
    # recursion carried the running logz as a standalone additive term, so a -1e300
    # sentinel dead point inflated h to ~1e300 and logzerr to ~1e141. The stable
    # recursion (logz only inside differences) must keep logzerr finite and sane.
    logl(x::AbstractVector) = x[1] < 0.6 ? -0.5 * sum(((xi - 0.3) / 0.1)^2 for xi in x) : -Inf
    priors = [Uniform(0.0, 1.0) for _ in 1:3]   # ~40% of init points are sentinels
    model = NestedModel(logl, priors)
    spl = Nested(3, 500; bounds=Bounds.MultiEllipsoid, proposal=Proposals.RWalk())

    _, state = sample(rng, model, spl; dlogz=0.1)
    @test isfinite(state.logz)
    @test isfinite(state.logzerr)
    @test 0 < state.logzerr < 10        # was ~1e141 before the fix
    @test isfinite(state.h) && state.h ≥ 0

    # parallel batch path shares the recursion — must also stay finite
    _, pstate = sample_parallel(rng, model, spl; batch_size=4, parallel=false, dlogz=0.1)
    @test isfinite(pstate.logzerr) && 0 < pstate.logzerr < 10
end

@testset "Stopping criterion" begin
    logl(x::AbstractVector) =  exp(-x[1]^2 / 2) / √(2π)
    priors = [Uniform(-1, 1)]
    model = NestedModel(logl, priors)
    spl = Nested(1, 500)
    
    chains, state = sample(rng, model, spl; add_live=false, dlogz=1.0)
    logz_remain = maximum(state.logl) + state.logvol
    delta_logz = logaddexp(state.logz, logz_remain) - state.logz
    @test delta_logz ≤ 1.0

    chains, state = sample(rng, model, spl; add_live=false, maxiter=3)
    @test state.it == 3

    chains, state = sample(rng, model, spl; add_live=false, maxcall=10)
    @test state.ncall == 10

    chains, state = sample(rng, model, spl; add_live=false, maxlogl=0.2)
    @test state.logl[1] ≥ 0.2
end
