import pandas as pd


def load_bbg_data(
    tickers: dict,
    start_date: str,
    end_date: str,
    frequency: str = "MONTHLY",
    bbg_client=None,
):
    """
    Generic Bloomberg-style data loader.

    This function avoids any proprietary API dependency.
    Pass a Bloomberg client with a .bdh() method if available.

    Expected client behavior:
    client.bdh(ticker, field, start_date, end_date, per=frequency)
    """

    if bbg_client is None:
        raise ValueError(
            "No Bloomberg client supplied. Use sample CSV data or pass a generic client."
        )

    frames = []

    for factor_name, cfg in tickers.items():
        ticker = cfg["ticker"]
        field = cfg.get("field", "PX_LAST")

        data = bbg_client.bdh(
            ticker,
            field,
            start_date,
            end_date,
            per=frequency,
        )

        data = data.rename(columns={field: factor_name})
        frames.append(data[[factor_name]])

    result = pd.concat(frames, axis=1)
    result.index.name = "Date"

    return result
