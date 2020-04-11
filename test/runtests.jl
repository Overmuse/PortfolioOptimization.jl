using Test, PortfolioOptimization

@testset "All tests" begin
    @testset "Weight construction" begin
        σ = [0.1, 0.2, 0.3, 0.4]
        ρ = [1.0  0.8  0.0  0.0;
             0.8  1.0  0.0  0.0;
             0.0  0.0  1.0 -0.5;
             0.0  0.0 -0.5  1.0]
        Σ = σ .* ρ .* σ'
        w_ew = optimize(EqualWeight(), Σ)
        w_erc = optimize(EqualRiskContribution(), Σ)
        w_mv = optimize(MinimumVariance(1/4), Σ)
        w_gmv = optimize(GlobalMinimumVariance(), Σ)
        w_md = optimize(MostDiversified(), Σ)

        @test portfolio_volatility(w_gmv, Σ) == mapreduce(w -> portfolio_volatility(w, Σ),
                                                          min,
                                                          [w_ew, w_erc, w_mv, w_gmv, w_md])

        @test diversification_ratio(w_gmv, Σ) == mapreduce(w -> diversification_ratio(w, Σ),
                                                           min,
                                                           [w_ew, w_erc, w_mv, w_gmv, w_md])

        @test w_ew == [1/4, 1/4, 1/4, 1/4]
        @test herfindahl(w_ew) == 1/4
        @test iszero(modified_herfindahl(w_ew))

        rc = risk_contribution(w_erc, Σ)
        @test all(y -> y ≈ first(rc), rc)
        @test risk_contribution_ratio(w_erc, Σ) ≈ [1/4, 1/4, 1/4, 1/4]

        mrc = marginal_risk_contribution(w_mv, Σ)
        @test all(y -> y ≈ first(mrc), mrc[.!iszero.(w_mv)])

        @test w_ew == optimize(MinimumVariance(1), Σ)
    end
    @testset "Discrete allocation" begin
        weights = [0.1, 0.2, 0.3, 0.4]
        prices = [100.0, 100.0, 100.0, 100.0]
        shares, cash_remaining = discrete_allocation(weights, prices, 1000.0)
        @test shares == [1, 2, 3, 4]
        @test cash_remaining == 0.0
        shares, cash_remaining = discrete_allocation(weights, prices, 1050.0)
        @test shares == [1, 2, 3, 4]
        @test cash_remaining == 50.0
    end
end
