# extract_clean.py

import pandas as pd
import numpy as np
import datetime as dt

def simulate_ohlcv(n=10000):
    ts = pd.date_range("2024-01-01", periods=n, freq="5T")
    price = np.cumsum(np.random.randn(n)) + 18000
    ohlc = pd.DataFrame({
        'timestamp': ts,
        'open': price + np.random.randn(n),
        'high': price + np.random.rand(n) * 10,
        'low': price - np.random.rand(n) * 10,
        'close': price,
        'volume': np.random.randint(100, 1000, n)
    })
    return ohlc

def simulate_option_chain(base_df):
    strikes = np.arange(17500, 18500, 50)
    expiries = pd.date_range("2024-01-04", periods=4, freq="W-THU")
    data = []
    for t in base_df['timestamp']:
        for strike in strikes:
            for ex in expiries:
                data.append({
                    'timestamp': t,
                    'expiry': ex,
                    'strike': strike,
                    'call_oi': np.random.randint(1000, 5000),
                    'put_oi': np.random.randint(1000, 5000)
                })
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = simulate_ohlcv()
    opt_df = simulate_option_chain(df)
    opt_agg = opt_df.groupby('timestamp')[['call_oi', 'put_oi']].sum().reset_index()
    df = df.merge(opt_agg, on='timestamp', how='left').fillna(method='ffill')
    df.to_csv("../data/processed/cleaned_data.csv", index=False)
