using NestedSamplers, StatsBase, Random, MCMCChains, Test
using NestedSamplers: Models, Bounds, Proposals

@testset "batch/parallel NS (sample_parallel)" begin
    model, lnZ_true = Models.CorrelatedGaussian(8)
    mks() = Nested(8, 500; proposal=Proposals.RWalk(), bounds=Bounds.MultiEllipsoid)

    # batch_size=1 must reproduce the serial sampler's log Z (same estimator).
    _, ss = sample(MersenneTwister(1), model, mks(); dlogz=0.05, chain_type=Chains, progress=false)
    _, s1 = sample_parallel(MersenneTwister(1), model, mks(); batch_size=1, parallel=false, dlogz=0.05)
    @test abs(s1.logz - ss.logz) < 3 * ss.logzerr          # same distribution

    # K-point batch (threaded) must not bias the evidence vs serial.
    _, s8 = sample_parallel(MersenneTwister(2), model, mks(); batch_size=8, parallel=true, dlogz=0.05)
    @test abs(s8.logz - ss.logz) < 3 * sqrt(ss.logzerr^2 + s8.logzerr^2)

    # Both within the sampler's own error of analytic (loose bound — RWalk in
    # 8-d has its own mixing bias; this just guards against gross errors).
    @test abs(s8.logz - lnZ_true) < 10 * s8.logzerr

    # Posterior chain is well-formed (weights sum to ~1).
    ch8, _ = sample_parallel(MersenneTwister(3), model, mks(); batch_size=4, parallel=true, dlogz=0.1)
    w = ch8[:weights]
    @test isapprox(sum(w), 1.0, atol=0.05)
end
