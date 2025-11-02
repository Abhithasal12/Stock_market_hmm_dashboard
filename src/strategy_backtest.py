import pandas as pd
import numpy as np

df = pd.read_csv("../data/regimes.csv", parse_dates=['timestamp'])

df['ema5'] = df['close'].ewm(span=5, adjust=False).mean()
df['ema15'] = df['close'].ewm(span=15, adjust=False).mean()
df['signal'] = 0
df.loc[(df['ema5'].shift(1) <= df['ema15'].shift(1)) & (df['ema5'] > df['ema15']), 'signal'] = 1
df.loc[(df['ema5'].shift(1) >= df['ema15'].shift(1)) & (df['ema5'] < df['ema15']), 'signal'] = -1

df['position'] = df['signal'].replace(to_replace=0, method='ffill')
df['strategy_ret'] = df['position'].shift(1) * df['return']
df['cum_pnl'] = (1 + df['strategy_ret']).cumprod()

summary = df.groupby('regime_label')['strategy_ret'].mean()
print(summary)

df.to_csv("../data/processed/strategy_results.csv", index=False)
