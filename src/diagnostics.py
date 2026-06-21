import numpy as np
import pandas as pd

from src.covariance import ewma_covariance, ewma_weights


def weighted_r_squared(
    y: pd.Series,
    X: pd.DataFrame,
    half_life: int = 12,
    add_intercept: bool = True,
) -> float:
    """
    EWMA-weighted R-squared.
    Useful for testing how much of each asset sleeve is explained by macro factors.
    """

    data = pd.concat([y, X], axis=1).dropna()

    y_clean = data.iloc[:, 0].values
    X_clean = data.iloc[:, 1:].values

    w = ewma_weights(len(data), half_life)

    if add_intercept:
        X_clean = np.column_stack([np.ones(len(X_clean)), X_clean])

    W = np.diag(w)

    beta = (
        np.linalg.pinv(X_clean.T @ W @ X_clean)
        @ (X_clean.T @ W @ y_clean)
    )

    y_hat = X_clean @ beta
    y_mean = np.average(y_clean, weights=w)

    ss_res = np.sum(w * (y_clean - y_hat) ** 2)
    ss_tot = np.sum(w * (y_clean - y_mean) ** 2)

    return 1 - ss_res / ss_tot


def r_squared_diagnostics(
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    half_life: int = 12,
) -> pd.DataFrame:
    """
    Calculates weighted R-squared for each asset sleeve.
    """

    results = {}

    for asset in asset_returns.columns:
        results[asset] = weighted_r_squared(
            y=asset_returns[asset],
            X=factor_returns,
            half_life=half_life,
            add_intercept=True,
        )

    return pd.DataFrame.from_dict(
        results,
        orient="index",
        columns=["Weighted R-Squared"],
    ).sort_values("Weighted R-Squared", ascending=False)


def stress_diagnostics(
    factor_returns: pd.DataFrame,
    factor_shock: pd.Series,
    half_life: int = 12,
):
    """
    Provides factor shock diagnostics:
    - EWMA volatility
    - shock z-score
    - Mahalanobis distance
    - contribution to stress severity
    """

    sigma_22 = ewma_covariance(factor_returns, half_life)

    vols = pd.Series(
        np.sqrt(np.diag(sigma_22)),
        index=sigma_22.index,
        name="EWMA Vol",
    )

    factor_shock = factor_shock.reindex(sigma_22.index)

    z_score = factor_shock / vols
    z_score.name = "Z Score"

    inv_corr = np.linalg.pinv(sigma_22)

    maha = float(np.sqrt(factor_shock.T @ inv_corr @ factor_shock))

    raw_contrib = pd.Series(
        factor_shock.values * (inv_corr @ factor_shock.values),
        index=factor_shock.index,
        name="Mahalanobis Contribution",
    )

    diagnostic_table = pd.DataFrame({
        "Shock": factor_shock,
        "EWMA Vol": vols,
        "Z Score": z_score,
        "Maha Contribution": raw_contrib,
        "Maha Contribution %": raw_contrib / raw_contrib.sum(),
    }).sort_values("Maha Contribution %", ascending=False)

    return {
        "factor_covariance": sigma_22,
        "z_scores": z_score,
        "mahalanobis_distance": maha,
        "diagnostic_table": diagnostic_table,
    }
