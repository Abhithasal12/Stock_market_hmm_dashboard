# train_lstm.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# Load data
df = pd.read_csv("../data/nifty/features_engineer_data.csv")

features = ['return', 'vol_30', 'oi_change']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df[features])
y = df['return'].shift(-1).fillna(0).values

seq_len = 60
X_seq, y_seq = [], []
for i in range(len(X_scaled) - seq_len):
    X_seq.append(X_scaled[i:i + seq_len])
    y_seq.append(y[i + seq_len])
X_seq, y_seq = np.array(X_seq), np.array(y_seq)

X_t = torch.tensor(X_seq, dtype=torch.float32)
y_t = torch.tensor(y_seq, dtype=torch.float32).view(-1, 1)
dataset = TensorDataset(X_t, y_t)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Model class
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

model = LSTMModel(X_seq.shape[2])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for xb, yb in dataloader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    epoch_loss /= len(dataset)
    print(f"Epoch {epoch + 1}/{num_epochs} Loss: {epoch_loss:.6f}")

# Save the trained model weights to a file
torch.save(model.state_dict(), '../data/raw/lstm_model.pth')
print("Model saved as lstm_model.pth")
