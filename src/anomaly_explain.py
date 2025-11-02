import pandas as pd
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
import shap

df = pd.read_csv("../data/processed/strategy_results.csv")

features = df[['return', 'vol_30', 'oi_change', 'strategy_ret']]
iso = IsolationForest(contamination=0.05, random_state=0)
iso.fit(features)
df['anom_score'] = -iso.decision_function(features)

df['trade_success'] = (df['strategy_ret'] > 0).astype(int)
clf = XGBClassifier().fit(features, df['trade_success'])
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(features)

shap.summary_plot(shap_values, features)
df.to_csv("../data/processed/anomaly_explain.csv", index=False)
