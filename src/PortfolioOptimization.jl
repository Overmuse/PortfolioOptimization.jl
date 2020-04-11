module PortfolioOptimization

using LinearAlgebra: diag
using StatsBase: cov2cor

export
      GlobalMinimumVariance,
      MinimumVariance,
      EqualRiskContribution,
      EqualWeight,
      MostDiversified,
      RiskParity,
      RichardRancolli,
      discrete_allocation,
      optimize,
      portfolio_volatility,
      herfindahl,
      modified_herfindahl,
      marginal_risk_contribution,
      risk_contribution,
      risk_contribution_ratio,
      diversification_ratio


include("stats.jl")
include("optimization.jl")

end
