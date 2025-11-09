# strategy_backtest.py

import pandas as pd
import numpy as np

# Load regimes and features data
df = pd.read_csv("../data/regimes.csv", parse_dates=['datetime'])

# Use the correct column for price (close_equity from your previous steps)
df['ema5'] = df['close_equity'].ewm(span=5, adjust=False).mean()
df['ema15'] = df['close_equity'].ewm(span=15, adjust=False).mean()

df['signal'] = 0
# Generate long/short signals on EMA crossovers with proper shift logic
df.loc[
    (df['ema5'].shift(1) <= df['ema15'].shift(1)) & (df['ema5'] > df['ema15']),
    'signal'
] = 1
df.loc[
    (df['ema5'].shift(1) >= df['ema15'].shift(1)) & (df['ema5'] < df['ema15']),
    'signal'
] = -1

# Forward-fill position, start with zero
df['position'] = df['signal'].replace(to_replace=0, method='ffill').fillna(0)

# Use the correct returns column (from feature engineering)
if 'return' not in df.columns:
    df['return'] = df['close_equity'].pct_change()
df['strategy_ret'] = df['position'].shift(1) * df['return'].fillna(0)
df['cum_pnl'] = (1 + df['strategy_ret']).cumprod()
df['strategy_ret']= df['strategy_ret'].fillna(0)
df['cum_pnl']= df['cum_pnl'].fillna(1)

# Performance summary by regime
summary = df.groupby('regime_label')['strategy_ret'].mean()
print(summary)

# Optional: Save detailed results
df.to_csv("../data/nifty/strategy_results.csv", index=False)
print("Strategy backtest complete. Results saved to ../data/nifty/strategy_results.csv")
