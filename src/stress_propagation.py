import numpy as np
import pandas as pd


def predict_asset_shocks(
    sigma_12: pd.DataFrame,
    sigma_22: pd.DataFrame,
    factor_shock: pd.Series,
) -> pd.Series:
    """
    Propagates macro factor shocks into asset return shocks.

    Formula:
    asset_shock = Cov(asset, factor) @ inv(Cov(factor)) @ factor_shock
    """

    factor_shock = factor_shock.reindex(sigma_22.index)

    asset_shock = sigma_12 @ np.linalg.pinv(sigma_22) @ factor_shock

    return pd.Series(
        asset_shock,
        index=sigma_12.index,
        name="Predicted Asset Shock",
    )


def portfolio_stress_return(
    asset_stress: pd.Series,
    strategic_weights: pd.Series,
) -> float:
    """
    Computes total portfolio stress return.
    """

    aligned_weights = strategic_weights.reindex(asset_stress.index).fillna(0)

    return float((aligned_weights * asset_stress).sum())


def portfolio_stress_contribution(
    asset_stress: pd.Series,
    strategic_weights: pd.Series,
) -> pd.DataFrame:
    """
    Computes contribution of each asset sleeve to total stress return.
    """

    aligned_weights = strategic_weights.reindex(asset_stress.index).fillna(0)

    contribution = aligned_weights * asset_stress

    return pd.DataFrame({
        "Weight": aligned_weights,
        "Asset Stress Return": asset_stress,
        "Stress Contribution": contribution,
        "Contribution %": contribution / contribution.sum()
    })
