import marimo

__generated_with = "0.11.21"
app = marimo.App(width="medium")


@app.cell
def _():
    import vectorbt as vbt
    import pandas as pd
    import numpy as np
    return np, pd, vbt


@app.cell
def _(pd, vbt):
    # Load historical close from CSV
    btc_1h = pd.read_csv('BTC_2024_1h_us.csv', index_col='Timestamp', parse_dates=True)
    btc_1D = pd.read_csv('BTC_daily_us.csv', index_col='Timestamp', parse_dates=True)

    # Extract close prices for 15m chart
    close = btc_1h['Close']

    # Compute indicators
    fast_ema = vbt.MA.run(close, 50, short_name='fast_ema')
    slow_ema = vbt.MA.run(close, 100, short_name='slow_ema')

    # Get previous day's high and low from daily close
    prev_high = btc_1D['High'].shift(1)
    prev_low = btc_1D['Low'].shift(1)

    # Long Entry signals
    cross_above_fast_ema = close.vbt.crossed_above(fast_ema.ma)
    price_above_slow_ema = close > slow_ema.ma
    entries = cross_above_fast_ema & price_above_slow_ema

    # Long Exit signals
    TP = prev_high.reindex(close.index, method='ffill')
    take_profit = close >= TP
    # Get entry price for each position
    entry_price = close[entries].reindex(close.index, method='ffill')
    RISK_DISTANCE = TP - entry_price  # Calculate risk based on entry price
    SL = entry_price - RISK_DISTANCE  # Stop loss is entry price minus risk distance
    stop_loss = close <= SL
    exits = take_profit | stop_loss

    # Short Entry signals
    # cross_below_fast_ema = close.vbt.crossed_below(fast_ema.ma)
    # price_below_slow_ema = close < slow_ema.ma
    # short_entries = cross_below_fast_ema & price_below_slow_ema

    # # Short Exit signals
    # SHORT_TP = prev_low.reindex(close.index, method='ffill')
    # short_take_profit = close <= SHORT_TP
    # SHORT_RISK_DISTANCE = close - SHORT_TP
    # SHORT_SL = close + SHORT_RISK_DISTANCE
    # short_stop_loss = close >= SHORT_SL
    # short_exits = short_take_profit | short_stop_loss

    # Simulate trades
    portfolio = vbt.Portfolio.from_signals(
        close=close, 
        entries=entries, 
        exits=exits,
        # short_entries=short_entries,
        # short_exits=short_exits,
        )


    portfolio.stats()
    return (
        RISK_DISTANCE,
        SL,
        TP,
        btc_1D,
        btc_1h,
        close,
        cross_above_fast_ema,
        entries,
        entry_price,
        exits,
        fast_ema,
        portfolio,
        prev_high,
        prev_low,
        price_above_slow_ema,
        slow_ema,
        stop_loss,
        take_profit,
    )


@app.cell
def _(portfolio):
    portfolio.plot().show()
    return


@app.cell
def _(SL, TP, close, entry_price, exits, pd):
    # Calculate exit prices
    exit_price = close[exits].reindex(close.index, method='ffill')

    # Create a DataFrame with all prices
    price_table = pd.DataFrame({
        'Close': close,
        'Entry_Price': entry_price,
        'Exit_Price': exit_price,
        'Take_Profit': TP,
        'Stop_Loss': SL
    })

    # Set display format for floats
    pd.set_option('display.float_format', lambda x: '%.2f' % x)

    # Display rows where we have either entry or exit signals
    filtered_table = price_table[
        price_table['Entry_Price'].notna() | 
        price_table['Exit_Price'].notna()
    ]

    # Add a column to show if it's an entry or exit point
    filtered_table['Signal'] = 'None'
    filtered_table.loc[filtered_table['Entry_Price'].notna(), 'Signal'] = 'Entry'
    filtered_table.loc[filtered_table['Exit_Price'].notna(), 'Signal'] = 'Exit'

    # Add columns to show if exit was due to TP or SL
    filtered_table['Exit_Type'] = 'None'
    tp_exits = (filtered_table['Exit_Price'] >= filtered_table['Take_Profit']) & (filtered_table['Exit_Price'].notna())
    sl_exits = (filtered_table['Exit_Price'] <= filtered_table['Stop_Loss']) & (filtered_table['Exit_Price'].notna())
    filtered_table.loc[tp_exits, 'Exit_Type'] = 'Take Profit'
    filtered_table.loc[sl_exits, 'Exit_Type'] = 'Stop Loss'

    filtered_table
    return exit_price, filtered_table, price_table, sl_exits, tp_exits


@app.cell
def _(portfolio):
    # Get all trades
    trades = portfolio.trades

    # Convert to DataFrame and display
    trades_df = trades.records_readable



    # Display trade statistics

    trades.stats()


    trades_df
    return trades, trades_df


@app.cell
def _(vbt):
    import matplotlib.pyplot as plt
    from mplfinance.original_flavor import candlestick_ohlc
    import matplotlib.dates as mdates
    from matplotlib.dates import num2date

    def plot_rsi(df, start_date='2024-01-01', end_date='2025-01-01'):
        plt.rcParams['figure.figsize'] = [20, 10]
        fig = plt.figure()
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        ax1 = plt.subplot2grid((5, 4), (0, 0), rowspan=4, colspan=4)

        # Prepare data for candlestick_ohlc
        df = df.copy()  # Create copy to avoid SettingWithCopyWarning
        df['Date'] = mdates.date2num(df.index)
        ohlc = df[['Date', 'Open', 'High', 'Low', 'Close']].to_numpy()

        candlestick_ohlc(ax1, ohlc, width=0.6, colorup='g', colordown='r')
        plt.title("BTC Price with RSI")
        plt.ylabel("Price")
        ax1.grid(True)

        # Calculate RSI
        rsi = vbt.RSI.run(df['Close'], window=14)

        # Plot RSI
        ax2 = plt.subplot2grid((5,4), (4,0), rowspan=1, colspan=4, sharex=ax1)
        ax2.fill_between(df.index, 30, 70, color="#eeeee4", alpha=0.5)
        ax2.plot(df.index, rsi.rsi, color='blue', linewidth=1)
        ax2.axhline(y=30, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        ax2.set_ylabel("RSI")
        ax2.grid(True)

        # Format x-axis
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.subplots_adjust(hspace=0)

        return fig
    return candlestick_ohlc, mdates, num2date, plot_rsi, plt


@app.cell
def _(pd, plot_rsi, vbt):
    # Download BTC data for 2024
    btc = vbt.YFData.download(
        "BTC-USD",  # Bitcoin symbol
        start='2024-01-01',
        end='2024-03-19',  # Current date
        missing_index='drop'  # Handle missing data
    ).get(["Open", "Close", "High", "Low", "Volume"])

    # Reset index to ensure Date is a column
    btc = btc.reset_index()

    # Convert Date column to datetime if not already
    btc['Date'] = pd.to_datetime(btc['Date'])

    # Set Date as index
    btc = btc.set_index('Date')

    # Save to CSV (optional)
    btc.to_csv('BTC_2024.csv')

    plot_rsi(btc)
    return (btc,)


@app.cell
def _(vbt, pd, np):
    def optimize_RSI_strategy(close, start_date='2024-01-01', end_date='2024-03-19'):
        # Filter data by date range
        mask = (close.index >= start_date) & (close.index <= end_date)
        close = close[mask]
        
        # Define parameter ranges for optimization
        rsi_window = np.arange(10, 30, 2)  # RSI periods from 10 to 30
        rsi_oversold = np.arange(20, 40, 5)  # Oversold levels from 20 to 40
        rsi_overbought = np.arange(60, 80, 5)  # Overbought levels from 60 to 80
        stop_loss_pct = np.arange(0.01, 0.05, 0.01)  # Stop loss from 1% to 5%
        take_profit_pct = np.arange(0.02, 0.10, 0.02)  # Take profit from 2% to 10%

        # Create custom RSI strategy indicator
        RSIStrategy = vbt.IndicatorFactory(
            class_name='RSIStrategy',
            short_name='rsi_strategy',
            input_names=['close'],
            param_names=['rsi_window', 'rsi_oversold', 'rsi_overbought', 'stop_loss_pct', 'take_profit_pct'],
            output_names=['entries', 'exits']
        ).from_apply_func(
            lambda close, rsi_window, rsi_oversold, rsi_overbought, stop_loss_pct, take_profit_pct: (
                # Calculate RSI
                rsi := vbt.RSI.run(close, window=rsi_window).rsi,
                
                # Generate entry signals
                entries := rsi < rsi_oversold,
                
                # Calculate stop loss and take profit levels
                entry_price := np.where(entries, close, np.nan),
                entry_price := entry_price.flatten(),  # Flatten to 1D
                entry_price := pd.Series(entry_price).fillna(method='ffill'),
                stop_loss := entry_price * (1 - stop_loss_pct),
                take_profit := entry_price * (1 + take_profit_pct),
                
                # Generate exit signals
                stop_loss_exit := close <= stop_loss,
                take_profit_exit := close >= take_profit,
                rsi_exit := rsi > rsi_overbought,
                
                # Combine exit signals
                exits := stop_loss_exit | take_profit_exit | rsi_exit,
                
                {'entries': entries, 'exits': exits}
            )[-1]
        )

        # Run optimization
        results = RSIStrategy.run(
            close,
            rsi_window=rsi_window,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            param_product=True
        )

        # Create portfolio for each parameter combination
        portfolios = vbt.Portfolio.from_signals(
            close=close,
            entries=results.entries,
            exits=results.exits,
            freq='1D'
        )

        # Get performance metrics for each portfolio
        metrics = pd.DataFrame({
            'total_return': portfolios.total_return(),
            'sharpe_ratio': portfolios.sharpe_ratio(),
            'max_drawdown': portfolios.max_drawdown(),
            'win_rate': portfolios.win_rate(),
            'profit_factor': portfolios.profit_factor()
        })

        # Find best strategy
        best_idx = metrics['total_return'].idxmax()
        best_params = results.params.iloc[best_idx]
        best_portfolio = portfolios[best_idx]
        
        return {
            'best_params': best_params,
            'best_metrics': metrics.iloc[best_idx].to_dict(),
            'portfolio': best_portfolio,
            'all_results': metrics
        }

    return optimize_RSI_strategy


@app.cell
def _(optimize_RSI_strategy, btc):
    # Run optimization
    results = optimize_RSI_strategy(btc['Close'])
    
    # Display best parameters and metrics
    print("Best Strategy Parameters:")
    print(results['best_params'])
    print("\nBest Strategy Metrics:")
    print(results['best_metrics'])
    
    # Plot the best strategy
    results['portfolio'].plot().show()
    
    return results


if __name__ == "__main__":
    app.run()
