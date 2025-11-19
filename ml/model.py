import torch
import joblib
import numpy as np
from torch import nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

MMSE_LIMIT = 30

# neural network: 75 -> 32 -> 16 -> 1
class MMSERegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(75, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# feature scaling
scaler = StandardScaler()
scaler_fitted = False

# initialize model, loss, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

regressor = MMSERegression().to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(regressor.parameters(), lr=1e-3, weight_decay=1e-5)

# ========== FEATURE SCALING ========== #

def fit_scaler(X):
    global scaler, scaler_fitted
    scaler.fit(X)
    scaler_fitted = True
    return scaler

def transform_features(X):
    if not scaler_fitted:
        raise RuntimeError("Scaler not fitted! Call fit_scaler() first.")
    return scaler.transform(X)

def save_scaler(fp: Path):
    joblib.dump(scaler, str(fp))

def load_scaler(fp: Path):
    global scaler, scaler_fitted
    scaler = joblib.load(str(fp))
    scaler_fitted = True

# ========== DATALOADER ========== #

# creates a dataloader from inputs
# X: [N, 75], y: [N]
def create_dataloader(X, y, batch_size=32, shuffle=True):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

# ========== TRAINING & TESTING ========== #

# train one batch
def train_step(x, y):
    x = x.to(device)
    y = y.to(device)

    optimizer.zero_grad()
    pred = regressor(x) * MMSE_LIMIT
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()

    return loss.item()

# train an epoch
def train_one_epoch(loader):
    total = 0.0
    count = 0

    regressor.train()

    for batch_x, batch_y in loader:
        loss = train_step(batch_x, batch_y)
        total += loss
        count += 1

    return total / count

# test with a test loader
def test(loader):
    regressor.eval()
    total_loss = 0.0
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred = regressor(batch_x) * MMSE_LIMIT
            loss = criterion(pred, batch_y)

            total_loss += loss.item()
            predictions.extend(pred.cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())

    mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
    return total_loss / len(loader), mae

# ========== SAVE/LOAD WEIGHTS ========== #

# save weights and biases
def save(fp: Path):
    torch.save(regressor.state_dict(), str(fp))

# load weights and biases
def load(fp: Path):
    regressor.load_state_dict(torch.load(str(fp), map_location=device))
