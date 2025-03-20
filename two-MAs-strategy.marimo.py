import marimo

__generated_with = "0.11.21"
app = marimo.App(width="medium")


@app.cell
def _():
    import vectorbt as vbt
    import numpy as np
    return np, vbt


@app.cell
def _(vbt):
    # Get data from yfinance
    symbols = ["BTC-USD", "ETH-USD", "LTC-USD"]
    price = vbt.YFData.download(symbols, missing_index='drop').get('Close')

    return price, symbols


@app.cell
def _(np, price, vbt):
    # Define stragety

    windows = np.arange(2, 101)
    fast_ma, slow_ma = vbt.MA.run_combs(price, window=windows, r=2, short_names=['fast', 'slow'])
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)

    pf_kwargs = dict(size=np.inf, fees=0.001, freq='1D')
    pf = vbt.Portfolio.from_signals(price, entries, exits, **pf_kwargs)
    return entries, exits, fast_ma, pf, pf_kwargs, slow_ma, windows


@app.cell
def _(pf):
    pf.save('two-MAs-strategy.pkl')
    return


@app.cell
def _():
    #pf = vbt.Porfolio.load('two-MAs-strategy.pkl')
    return


@app.cell
def _(pf):
    fig = pf.total_return().vbt.heatmap(
        x_level='fast_window', y_level='slow_window', slider_level='symbol', symmetric=True,
        trace_kwargs=dict(colorbar=dict(title='Total return', tickformat='%')))
    fig.show()
    return (fig,)


if __name__ == "__main__":
    app.run()
