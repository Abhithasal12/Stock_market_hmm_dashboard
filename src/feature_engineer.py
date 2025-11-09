import pandas as pd
import numpy as np

# Load merged, cleaned data
df = pd.read_csv("../data/nifty/cleaned_nifty_data_all_1.csv", parse_dates=['datetime'])

# Price-based features (on equity)
df['return'] = df['close_equity'].pct_change()
df['log_return'] = np.log1p(df['return'])
df['vol_30'] = df['return'].rolling(30).std()
df['atr_14'] = (df['high_equity'] - df['low_equity']).rolling(14).mean()

# Option open interest feature
df['oi_change'] = df['call_open_interest'].diff().fillna(0) + df['put_open_interest'].diff().fillna(0)

# Futures basis (spread between futures and equity)
df['basis'] = df['close_futures'] - df['close_equity']

# Volume-to-open interest ratio for equity and futures
df['vol_oi_ratio_equity'] = df['volume_equity'] / (df['open_interest_equity'] + 1)
df['vol_oi_ratio_futures'] = df['volume_futures'] / (df['open_interest_futures'] + 1)

# Lag features for ML
df['close_equity_lag_1'] = df['close_equity'].shift(1)
df['close_equity_lag_5'] = df['close_equity'].shift(5)
df['return_lag_1'] = df['return'].shift(1)
df['return_lag_5'] = df['return'].shift(5)

# Drop rows with any NaNs caused by rolling and shifting
df.dropna(inplace=True)

# Save features
df.to_csv("../data/nifty/features_engineer_data.csv", index=False)

print("Feature engineering completed. Features saved to ../data/nifty/features_engineer_data.csv")
