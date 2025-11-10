Project Overview This project builds a full quantitative trading analytics pipeline:

Market data extraction (OHLCV, futures, options with OI)

Data cleaning & feature engineering

Regime detection using Hidden Markov Models (HMM)

5/15 EMA crossover strategy backtest

LSTM model for predictive scoring

Anomaly detection and SHAP-based explainability

Interactive Power BI dashboard for reporting

├── dashboard/ # Power BI .pbix and supporting visuals ├── data/ # Raw and processed CSV data (market, features, trades, regimes, SHAP) ├── src/ # Python scripts for ETL, feature engineering, modeling, backtest, explainability ├── Market_Data_Pipeline_Report.pdf # Final findings and summary ├── requirements.txt # Python dependencies ├── README.md # This file

How to Run Install Python Dependencies

Open your terminal and run:

bash pip install -r requirements.txt Run Data Pipeline Scripts

In the src/ directory, run the scripts in this order:

text python extract_clean.py python feature_engineer.py python hmm_regimes.py python strategy_backtest.py python train_lstm.py python anomaly_explain.py All outputs are saved in data/ for analysis and dashboard input.

Power BI Dashboard

Open the .pbix file in the dashboard/ folder using Power BI Desktop.

Click “Refresh” to load latest CSVs.

Explore price/EMA charts, trade logs, performance KPIs, regime filtering, and option chain heatmaps.

Results and Summary

See Market_Data_Pipeline_Report.pdf for model results, KPIs, regime analysis and recommendations.

Main Features Accurate 5-min data simulation and alignment (with OI, futures, options)

Outlier removal and rolling features

Regime detection and labeling (Uptrend, Sideways, Downtrend)

EMA crossover strategy with realistic backtest

LSTM returns prediction (walk-forward validation)

Trade anomaly detection and feature importance (SHAP)

Power BI dashboard for interactive data exploration

Requirements Python 3.8+

Power BI Desktop

All dependencies in requirements.txt

Author Abhishek Thasal