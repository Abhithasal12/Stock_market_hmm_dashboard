import pandas as pd
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
import numpy as np

df = pd.read_csv("../data/processed/features.csv")
X = df[['vol_30', 'return', 'oi_change']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

hmm = GaussianHMM(n_components=3, n_iter=200, covariance_type='full', random_state=42)
hmm.fit(X_scaled)
df['regime'] = hmm.predict(X_scaled)

means = df.groupby('regime')['return'].mean().sort_values()
mapping = {means.index[0]:'DOWNTREND', 
           means.index[1]:'SIDEWAYS', 
           means.index[2]:'UPTREND'}
df['regime_label'] = df['regime'].map(mapping)

df.to_csv("../data/regimes.csv", index=False)
