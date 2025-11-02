import pandas as pd
import numpy as np

df = pd.read_csv("../data/processed/cleaned_data.csv", parse_dates=['timestamp'])

df['return'] = df['close'].pct_change()
df['log_return'] = np.log1p(df['return'])
df['vol_30'] = df['return'].rolling(30).std()
df['atr_14'] = (df['high'] - df['low']).rolling(14).mean()
df['oi_change'] = df['call_oi'].diff() + df['put_oi'].diff()

df.dropna(inplace=True)
df.to_csv("../data/processed/features.csv", index=False)
