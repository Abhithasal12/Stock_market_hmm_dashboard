#anomaly_explain.py

import pandas as pd
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

# Load the strategy results data
df = pd.read_csv("../data/nifty/strategy_results.csv")

# Choose features for anomaly detection and modeling
feature_cols = ['return', 'vol_30', 'oi_change', 'strategy_ret']
features = df[feature_cols]

# Fit Isolation Forest for anomaly scores
iso = IsolationForest(contamination=0.05, random_state=0)
iso.fit(features)
df['anom_score'] = -iso.decision_function(features)  # Higher means more anomalous

# Define binary target: trade success (positive strategy return)
df['trade_success'] = (df['strategy_ret'] > 0).astype(int)

# Train an XGBoost classifier (adjust hyperparameters as needed)
clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
clf.fit(features, df['trade_success'])

# SHAP explainability
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(features)

# Summary plot for global feature importance
shap.summary_plot(shap_values, features, show=False)
plt.tight_layout()
plt.savefig('../data/raw/shap_summary_plot.png')  # Save plot to file
plt.close()

# Save dataframe with anomaly scores and explanations
df.to_csv("../data/nifty/anomaly_explain.csv", index=False)
print("Anomaly detection and SHAP explanations saved.")
