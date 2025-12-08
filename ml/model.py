import torch
import joblib
import numpy as np
from torch import nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# neural network:
#   A [# -> 64], L [# -> 32], S [# -> 32]
#   A+L+S [128 -> 64 -> 32 -> 1]
class MMSERegression(nn.Module):
    def __init__(self, n_acoustics, n_linguistics, n_semantics, dropout=0.3):
        super().__init__()

        # save feature type count
        self.n_acoustics = n_acoustics
        self.n_linguistics = n_linguistics
        self.n_semantics = n_semantics

        # separate encoders for each feature type
        self.acoustics_encoder = nn.Sequential(
            nn.Linear(n_acoustics, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.linguistics_encoder = nn.Sequential(
            nn.Linear(n_linguistics, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.semantics_encoder = nn.Sequential(
            nn.Linear(n_semantics, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # everything together
        self.fusion = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        acoustic_features = x[:, :self.n_acoustics]
        linguistic_features = x[:, self.n_acoustics:self.n_acoustics + self.n_linguistics]
        semantic_features = x[:, self.n_acoustics + self.n_linguistics:]

        # encode each feature type
        acoustic_emb = self.acoustics_encoder(acoustic_features)
        linguistic_emb = self.linguistics_encoder(linguistic_features)
        semantic_emb = self.semantics_encoder(semantic_features)

        # concatenate and fuse (scale to mmse limit [0, 30])
        input_vec = torch.cat([acoustic_emb, linguistic_emb, semantic_emb], dim=1)
        output = self.fusion(input_vec)
        output_clamped = torch.clamp(output, 0, 30)

        return output_clamped

# feature scaling
scaler = StandardScaler()
scaler_fitted = False

# initialize model, loss, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

regressor = MMSERegression(
    n_acoustics=52,
    n_linguistics=29,
    n_semantics=18
).to(device)
criterion = nn.HuberLoss(delta=1.5)
optimizer = torch.optim.Adam(regressor.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3
)

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
# X: [N, #], y: [N]
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
    pred = regressor(x)
    loss = criterion(pred, y)
    loss.backward()

    # gradient clipping just to be safe
    torch.nn.utils.clip_grad_norm_(regressor.parameters(), max_norm=1.0)

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

            pred = regressor(batch_x)
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
