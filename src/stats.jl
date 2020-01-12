export portfolio_volatility,
       herfindahl,
       modified_herfindahl,
       marginal_risk_contribution,
       risk_contribution,
       risk_contribution_ratio,
       diversification_ratio

function herfindahl(w)
    sum(x^2 for x in w)
end

function modified_herfindahl(w)
    n = length(w)
    (herfindahl(w) - 1/n) / (1 - 1/n)
end

function portfolio_volatility(w, Σ)
    √(w' * Σ * w)
end

function marginal_risk_contribution(w, Σ)
    Σ * w / portfolio_volatility(w, Σ)
end

function risk_contribution(w, Σ)
    w .* marginal_risk_contribution(w, Σ)
end

function risk_contribution_ratio(w, Σ)
    rc = risk_contribution(w, Σ)
    rc ./ sum(rc)
end

function diversification_ratio(w, Σ)
    w' * diag(Σ) / portfolio_volatility(w, Σ)
end
