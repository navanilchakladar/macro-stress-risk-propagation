import pandas as pd

from src.portfolio import (
    load_portfolio_returns,
    load_factor_returns,
    example_strategic_weights,
)
from src.covariance import build_partitioned_covariance
from src.stress_propagation import (
    predict_asset_shocks,
    portfolio_stress_return,
    portfolio_stress_contribution,
)
from src.diagnostics import r_squared_diagnostics, stress_diagnostics


def main():

    asset_returns = load_portfolio_returns(
        "data/sample_portfolio_returns.csv"
    )

    factor_returns = load_factor_returns(
        "data/sample_macro_factors.csv"
    )

    half_life = 12

    (
        sigma,
        sigma_11,
        sigma_12,
        sigma_21,
        sigma_22,
    ) = build_partitioned_covariance(
        asset_returns=asset_returns,
        factor_returns=factor_returns,
        half_life=half_life,
    )

    factor_shock = pd.Series({
        "Global_Equity": -0.10,
        "Global_Bond": -0.05,
        "Inflation": 0.02,
        "Credit_Spread": 0.02,
        "Commodity": 0.05,
    })

    asset_stress = predict_asset_shocks(
        sigma_12=sigma_12,
        sigma_22=sigma_22,
        factor_shock=factor_shock,
    )

    strategic_weights = example_strategic_weights()

    port_stress_return = portfolio_stress_return(
        asset_stress=asset_stress,
        strategic_weights=strategic_weights,
    )

    port_contribution = portfolio_stress_contribution(
        asset_stress=asset_stress,
        strategic_weights=strategic_weights,
    )

    r2_table = r_squared_diagnostics(
        asset_returns=asset_returns,
        factor_returns=factor_returns,
        half_life=half_life,
    )

    diagnostics = stress_diagnostics(
        factor_returns=factor_returns,
        factor_shock=factor_shock,
        half_life=half_life,
    )

    print("\nPredicted Asset Shocks")
    print(asset_stress)

    print("\nPortfolio Stress Return")
    print(port_stress_return)

    print("\nPortfolio Stress Contribution")
    print(port_contribution)

    print("\nWeighted R-Squared Diagnostics")
    print(r2_table)

    print("\nStress Diagnostics")
    print(diagnostics["diagnostic_table"])


if __name__ == "__main__":
    main()
