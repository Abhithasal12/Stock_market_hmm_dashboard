import pandas as pd
from pathlib import Path 

# Paths to your sample folders or just single sample files
equity_file = '../drive/sample/New folder/sample_2022_nov_equity_data.csv'
futures_file = '../drive/sample/New folder/sample_2022_nov_future_data.csv'
options_file = '../drive/sample/New folder/sample_2022_nov_option_data.csv'

# Load and clean equity data
equity_df = pd.read_csv(equity_file, parse_dates=['datetime'])
equity_df.dropna(how='all', inplace=True)                       # Remove completely blank rows
equity_df.dropna(subset=['datetime'], inplace=True)             # Keep only rows with datetime values
equity_df.sort_values('datetime', inplace=True)
equity_df.fillna(0, inplace=True)                                   # Fill remaining NaNs with 0
# Load and clean futures data
futures_df = pd.read_csv(futures_file, parse_dates=['datetime'])
futures_df.dropna(how='all', inplace=True)
futures_df.dropna(subset=['datetime'], inplace=True)
futures_df.sort_values('datetime', inplace=True)
futures_df.fillna(0, inplace=True)
# Load and clean options data
options_df = pd.read_csv(options_file, parse_dates=['datetime'])
options_df.dropna(how='all', inplace=True)
options_df.dropna(subset=['datetime'], inplace=True)
options_df.sort_values('datetime', inplace=True)
options_df.fillna(0, inplace=True)
# Aggregate options open interest (replace column names if needed)
opt_agg = options_df.groupby('datetime')[['call_open_interest', 'put_open_interest']].sum().reset_index()

# Merge equity and futures on datetime
merged_df = equity_df.merge(futures_df, on='datetime', how='inner', suffixes=('_equity', '_futures'))

# Merge aggregated options data on datetime
merged_df = merged_df.merge(opt_agg, on='datetime', how='inner')

# Fill any missing values using forward fill (useful for time series continuity)
merged_df.fillna(method='ffill', inplace=True)

# Save cleaned, merged dataset
output_path = '../data/nifty/cleaned_nifty_data_all_1.csv'
merged_df.to_csv(output_path, index=False)

print(f"Data cleaning done - saved at: {output_path}")
print(f"Final shape: {merged_df.shape}")