# ─────────────────────────────────────────────────────────────────────────────
# model.py  —  LSTM Model · Dataset · Training & Prediction Pipeline
#
# Imported by app.py:
#   from model import train_and_predict
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader


# ──────────────────────────────────────────────
# Architecture
# ──────────────────────────────────────────────

class LSTMModel(nn.Module):
    """
    Two-layer stacked LSTM with dropout, followed by a single linear
    output head that predicts the next normalised price step.

    Input shape  : (batch, seq_len, 1)
    Output shape : (batch, 1)
    """

    def __init__(self, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)          # (batch, seq, hidden)
        return self.fc(out[:, -1, :])  # last time-step → scalar


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class SeqDataset(Dataset):
    """
    Sliding-window dataset over a 1-D normalised price array.

    Each sample is a (seq_len,) input window and the immediate
    next value as the target label.
    """

    def __init__(self, X: np.ndarray, seq_len: int):
        self.X       = torch.tensor(X, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.X) - self.seq_len

    def __getitem__(self, i: int):
        return self.X[i : i + self.seq_len], self.X[i + self.seq_len]


# ──────────────────────────────────────────────
# Training + Inference
# ──────────────────────────────────────────────

def train_and_predict(
    prices: np.ndarray,
    future_days: int = 30,
    seq_len: int | None = None,
    epochs: int = 30,
    lr: float = 0.001,
    batch_size: int = 32,
    train_split: float = 0.85,
) -> np.ndarray:
    """
    Normalise `prices`, train an LSTM on the training split, then
    auto-regressively forecast `future_days` steps ahead.

    Parameters
    ----------
    prices       : 1-D array of historical close prices
    future_days  : number of business days to forecast
    seq_len      : look-back window (auto-sized to min(30, len//4) if None)
    epochs       : training epochs
    lr           : Adam learning rate
    batch_size   : DataLoader batch size
    train_split  : fraction of data used for training

    Returns
    -------
    np.ndarray of shape (future_days,) in original price scale
    """

    SEQ_LEN = seq_len if seq_len is not None else min(30, len(prices) // 4)

    # ── Normalise ──
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()

    # ── Build DataLoader ──
    split  = int(len(scaled) * train_split)
    tr_ds  = SeqDataset(scaled[:split], SEQ_LEN)
    tr_dl  = DataLoader(tr_ds, batch_size=batch_size, shuffle=False)

    # ── Initialise model ──
    model   = LSTMModel()
    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # ── Training loop ──
    for _ in range(epochs):
        model.train()
        for xb, yb in tr_dl:
            opt.zero_grad()
            pred = model(xb.unsqueeze(-1)).squeeze()
            loss = loss_fn(pred, yb.squeeze())
            loss.backward()
            opt.step()

    # ── Auto-regressive forecasting ──
    last_seq   = scaled[-SEQ_LEN:].tolist()
    future_raw = []

    model.eval()
    with torch.no_grad():
        for _ in range(future_days):
            x   = torch.tensor(last_seq[-SEQ_LEN:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            out = model(x).item()
            future_raw.append(out)
            last_seq.append(out)

    # ── Inverse-transform back to price scale ──
    return scaler.inverse_transform(
        np.array(future_raw).reshape(-1, 1)
    ).flatten()
