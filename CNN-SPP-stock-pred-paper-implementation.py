import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------
# Config (paper-like choices)
# -----------------------------
WINDOW = 16          # sliding window length
HORIZON = 6          # predict 6 steps ahead
BATCH_SIZE = 128
EPOCHS = 15
LR = 1e-3
SYMBOL = "BTC-USD"
INTERVAL = "1h"

# -----------------------------
# Download BTC data
# -----------------------------
df = yf.download(SYMBOL, interval=INTERVAL, period="2y")
df = df[['Open','High','Low','Close','Volume']].dropna()

# -----------------------------
# Build sliding windows
# -----------------------------
X, y = [], []
close = df['Close'].values

for i in range(len(df) - WINDOW - HORIZON):
    window = df.iloc[i:i+WINDOW].values.T  # (channels, time)
    future_ret = close[i+WINDOW+HORIZON] / close[i+WINDOW] - 1
    label = 1 if future_ret > 0 else 0
    X.append(window)
    y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)

# -----------------------------
# Train / Test split (time-based)
# -----------------------------
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -----------------------------
# Dataset
# -----------------------------
class TSData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

train_loader = DataLoader(TSData(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(TSData(X_test, y_test),  batch_size=BATCH_SIZE)

# -----------------------------
# 1-D CNN (paper-style)
# -----------------------------
class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(5, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64,128,3,padding=1)
        self.conv3 = nn.Conv1d(128,256,3,padding=1)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.fc    = nn.Linear(256, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNN1D().to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

# -----------------------------
# Train
# -----------------------------
for epoch in range(EPOCHS):
    model.train()
    losses = []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"Epoch {epoch+1}/{EPOCHS} | Train loss: {np.mean(losses):.4f}")

# -----------------------------
# Evaluate
# -----------------------------
model.eval()
preds, trues = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        preds.extend(logits.argmax(1).cpu().numpy())
        trues.extend(yb.numpy())

acc = accuracy_score(trues, preds)
cm  = confusion_matrix(trues, preds)

print("\n=== BTC/USD CNN Results ===")
print(f"Accuracy: {acc:.4f}")
print("Confusion Matrix:")
print(cm)
