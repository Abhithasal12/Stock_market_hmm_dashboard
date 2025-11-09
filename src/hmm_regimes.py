import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import numpy as np

# Load features (use correct file path if needed)
df = pd.read_csv("../data/nifty/features_engineer_data.csv")

# Features for regime detection
X = df[['vol_30', 'return', 'oi_change']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Gaussian Hidden Markov Model
hmm = GaussianHMM(n_components=3, n_iter=200, covariance_type='full', random_state=42)
hmm.fit(X_scaled)
df['regime'] = hmm.predict(X_scaled)

# Robust trend label mapping based on 'return' mean per cluster
means = df.groupby('regime')['return'].mean().sort_values()
labels = ['DOWNTREND', 'SIDEWAYS', 'UPTREND']
mapping = {reg_idx: labels[i] for i, reg_idx in enumerate(means.index)}
df['regime_label'] = df['regime'].map(mapping)

# Save results
df.to_csv("../data/regimes.csv", index=False)
print("Regime detection complete. Output saved to ../data/regimes.csv")

# Optional: Print regime distribution & mean returns
print(df['regime_label'].value_counts())
print(means)
