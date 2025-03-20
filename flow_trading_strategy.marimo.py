import marimo

__generated_with = "0.11.21"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as пр
    import matplotlib.pyplot as plt
    import vectorbt as vbt
    import warnings
    warnings. filterwarnings ("ignore")
    return pd, plt, vbt, warnings, пр


@app.cell
def _(vbt):
    # Download historical price data for TLT ETF from Yahoo Finance and extract the closing prices
    tlt = vbt.YFData. download ("BTC",
        start="2004-01-01",
        end="2024-12-01"
        ).get("Close" ).to_frame ( )
    close = tlt.Close
    return close, tlt


@app.cell
def _(tlt):
    tlt.to_csv('BTC_1D_2024.csv')
    # tlt = pd.read_csv('TLT_1D_2024.csv', index_col="Date", parse_dates=True)

    return


@app.cell
def _(tlt):
    close = tlt.Close
    return (close,)


@app.cell
def _(close, pd, tlt):
    # Set up empty dataframes to hold trading signals for short and long positions
    short_entries = pd.DataFrame.vbt.signals.empty_like(close)
    short_exits = pd.DataFrame.vbt.signals.empty_like(close)
    long_entries = pd.DataFrame.vbt.signals.empty_like(close)
    long_exits = pd.DataFrame.vbt.signals.empty_like(close)
    # Generate short entry signals on the first day of each new month
    short_entry_mask = ~tlt.index.tz_convert(None).to_period("M" ).duplicated ( )
    short_entries.iloc[short_entry_mask] = True
    # Generate short exit signals five days after short entry
    short_exit_mask = short_entries.shift(5).fillna (False)
    short_exits.iloc[short_exit_mask] = True
    # Generate long entry signals seven days before the end of each month
    long_entry_mask = short_entries.shift(-7).fillna (False)
    long_entries.iloc[long_entry_mask] = True
    # Generate long exit signals one day before the end I of each month
    long_exit_mask = short_entries.shift(-1).fillna(False)
    long_exits.iloc[long_exit_mask] = True
    return (
        long_entries,
        long_entry_mask,
        long_exit_mask,
        long_exits,
        short_entries,
        short_entry_mask,
        short_exit_mask,
        short_exits,
    )


@app.cell
def _(close, long_entries, long_exits, short_entries, short_exits, vbt):
    # Run the simulation and calculate the Sharpe ratio for the trading strategy
    pf = vbt.Portfolio.from_signals(close=close,
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        freq="1d")

    # Generate a plot with the strategy's performance.
    pf.plot().show()
    return (pf,)


@app.cell
def _(pf):
    pf.stats()
    return


if __name__ == "__main__":
    app.run()
