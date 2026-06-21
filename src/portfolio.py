import pandas as pd


def load_portfolio_returns(file_path: str) -> pd.DataFrame:
    """
    Loads asset sleeve returns.

    Expected format:
    Date, Equity, Bonds, Commodities
    """

    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    return df


def load_factor_returns(file_path: str) -> pd.DataFrame:
    """
    Loads macro factor returns.

    Expected format:
    Date, Global_Equity, Global_Bond, Inflation, Credit_Spread, Commodity
    """

    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    return df


def example_strategic_weights() -> pd.Series:
    return pd.Series({
        "Equity": 0.30,
        "Bonds": 0.50,
        "Commodities": 0.20,
    })
