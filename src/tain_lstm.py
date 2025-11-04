# tain_lstm.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("../data/processed/features.csv")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df[['return', 'vol_30', 'oi_change']])
y = df['return'].shift(-1).fillna(0).values

X_seq, y_seq = [], []
seq_len = 60
for i in range(len(X_scaled) - seq_len):
    X_seq.append(X_scaled[i:i+seq_len])
    y_seq.append(y[i+seq_len])
X_seq, y_seq = np.array(X_seq), np.array(y_seq)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

model = LSTMModel(X_seq.shape[2])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

X_t = torch.tensor(X_seq, dtype=torch.float32)
y_t = torch.tensor(y_seq, dtype=torch.float32).view(-1, 1)

for epoch in range(10):
    optimizer.zero_grad()
    preds = model(X_t)
    loss = criterion(preds, y_t)
    loss.backward()
    optimizer.step()
    print("Epoch", epoch, "Loss:", loss.item())
