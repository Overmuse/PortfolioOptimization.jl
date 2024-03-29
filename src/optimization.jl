abstract type AbstractPortfolioOptimizationTarget end
struct GlobalMinimumVariance <: AbstractPortfolioOptimizationTarget end
struct MinimumVariance <: AbstractPortfolioOptimizationTarget
    c
end
struct EqualWeight <: AbstractPortfolioOptimizationTarget end
struct EqualRiskContribution <: AbstractPortfolioOptimizationTarget end
struct MostDiversified <: AbstractPortfolioOptimizationTarget end
struct RiskParity <: AbstractPortfolioOptimizationTarget end
struct RichardRancolli <: AbstractPortfolioOptimizationTarget
    gmv
    mdp
    h
    erc
end
function _ccd(x, σ, ρ, λ_gmv, λ_mdp, λ_h, λ_erc, i)
    summation = sum(x[j] * ρ[i, j] * σ[j] for j in 1:length(x) if i != j)
    term₁ = (λ_gmv+λ_mdp*σ[i]-σ[i]*summation) / (2*(σ[i]^2+2λ_h))
    term₂ = √((σ[i] * summation - λ_gmv - λ_mdp * σ[i])^2 + 4(σ[i]^2+2λ_h)*λ_erc) / (2*(σ[i]^2+2λ_h))
    term₁ + term₂
end

function ccd!(x, σ, ρ, λ_gmv, λ_mdp, λ_h, λ_erc; iter = 100)
    for it in 1:iter
        for i in 1:length(x)
            x[i] = _ccd(x, σ, ρ, λ_gmv, λ_mdp, λ_h, λ_erc, i)
        end
    end
    x ./ sum(x)
end

function ccd(Σ, λ_gmv, λ_mdp, λ_h, λ_erc; iter = 100)
    num_assets = size(Σ, 1)
    ccd!(1 ./ (ones(eltype(Σ), num_assets)), sqrt.(diag(Σ)), cov2cor(Σ, sqrt.(diag(Σ))), λ_gmv, λ_mdp, λ_h, λ_erc, iter = iter)
end

function optimize(::GlobalMinimumVariance, Σ)
    I = ones(eltype(Σ), size(Σ, 1))
    Σ⁻¹ = inv(Σ)
    Σ⁻¹ * I / (I' * Σ⁻¹ * I)
end

function optimize(x::MinimumVariance, Σ)
    ccd(Σ, 1, 0, (x.c - 1/size(Σ, 1)) / (1 - x.c + eps()), 0)
end

function optimize(::EqualWeight, Σ)
    ones(size(Σ, 1)) ./ size(Σ, 1)
end

function optimize(::EqualRiskContribution, Σ)
    ccd(Σ, 0, 0, 0, 1)
end

function optimize(::MostDiversified, Σ)
    ccd(Σ, 0, 1, 0, 0)
end

function optimize(::RiskParity, Σ)
    ccd(Σ, 0, -1/eps(), 0, 1/eps())
end

function optimize(x::RichardRancolli, Σ)
    ccd(Σ, x.gmv, x.mdp, x.h, x.erc)
end

function discrete_allocation(target_weights, prices, cash)
    cash_remaining = cash
    shares_bought = zeros(Int, length(target_weights))
    for idx in sortperm(target_weights, rev = true)
        price = prices[idx]
        shares = target_weights[idx] * cash ÷ price
        cost = shares * price
        if cost > cash_remaining
            shares = cash_remaining ÷ price
        end
        cash_remaining -= cost
        shares_bought[idx] = shares
    end
    while cash_remaining > 0
        current_weights = prices .* shares_bought / (prices' * shares_bought)
        deficit = target_weights .- current_weights
        idx = argmax(deficit)
        weight = target_weights[idx]
        price = prices[idx]
        while price > cash_remaining
            deficit[idx] = 0
            idx = argmax(deficit)
            if deficit[idx] <= 0
                break
            end
            weight = target_weights[idx]
            price = prices[idx]
        end
        if deficit[idx] <= 0
            break
        end
        shares_bought[idx] += 1
        cash_remaining -= price
    end
    return shares_bought, cash_remaining
end
