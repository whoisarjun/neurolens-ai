# Model center

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# neural network:
#   A [52 -> 64], L [29 -> 32], S [18 -> 32], E [1024 -> 16]
#   A+L+S [144 -> 256 -> 64 -> 32 -> specific output head]
class Backbone(nn.Module):
    def __init__(self, n_acoustics, n_linguistics, n_semantics, n_embeddings):
        super().__init__()

        # save feature type count
        self.n_acoustics = n_acoustics
        self.n_linguistics = n_linguistics
        self.n_semantics = n_semantics
        self.n_embeddings = n_embeddings

        # separate encoders for each feature type
        self.acoustics_encoder = nn.Sequential(
            nn.Linear(n_acoustics, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        ) if n_acoustics > 0 else lambda n: n
        self.linguistics_encoder = nn.Sequential(
            nn.Linear(n_linguistics, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.3)
        ) if n_linguistics > 0 else lambda n: n
        self.semantics_encoder = nn.Sequential(
            nn.Linear(n_semantics, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.3)
        ) if n_semantics > 0 else lambda n: n
        self.embeddings_encoder = nn.Sequential(
            nn.Linear(n_embeddings, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        ) if n_embeddings > 0 else lambda n: n

        # everything together
        in_features = (64 if n_acoustics > 0 else 0) + (32 if n_linguistics > 0 else 0) + (32 if n_semantics > 0 else 0) + (64 if n_embeddings > 0 else 0)
        self.fusion = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        acoustic_features = x[:, :self.n_acoustics]
        linguistic_features = x[:, self.n_acoustics:self.n_acoustics + self.n_linguistics]
        semantic_features = x[:, self.n_acoustics + self.n_linguistics:self.n_acoustics + self.n_linguistics + self.n_semantics]
        embeddings_features = x[:, self.n_acoustics + self.n_linguistics + self.n_semantics:]

        # encode each feature type
        acoustic_emb = self.acoustics_encoder(acoustic_features)
        linguistic_emb = self.linguistics_encoder(linguistic_features)
        semantic_emb = self.semantics_encoder(semantic_features)
        embeddings_emb = self.embeddings_encoder(embeddings_features)

        # concatenate and fuse (scale to mmse limit [0, 30])
        input_vec = torch.cat([acoustic_emb, linguistic_emb, semantic_emb, embeddings_emb], dim=1)
        output = self.fusion(input_vec)  # -> send to output head

        return output

# mmse regression [1123 -> backbone -> 32 -> 1]
class MMSERegression(nn.Module):
    def __init__(self, backbone: Backbone):
        super().__init__()
        self.backbone = backbone
        self.net = nn.Sequential(
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.net(features)
        return torch.clamp(output, 0, 30)

# cog status classification [1123 -> backbone -> 32 -> 3 logits]
class CognitiveStatusClassification(nn.Module):
    def __init__(self, backbone: Backbone):
        super().__init__()
        self.backbone = backbone
        self.net = nn.Sequential(
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.Dropout(0.3),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        features = self.backbone(x)
        output = self.net(features)
        return output

cog_statuses = ['HC', 'MCI', 'AD']

# feature scaling
scaler = StandardScaler()
scaler_fitted = False

# initialize model, loss, optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def new_backbone(n_acoustics=52, n_linguistics=29, n_semantics=18, n_embeddings=1024):
    return Backbone(n_acoustics, n_linguistics, n_semantics, n_embeddings).to(device)

def new_multitask(backbone: Backbone, lr=1e-3, weight_decay=1e-5):
    # creates joint regressor and classifier with backbone
    regressor = MMSERegression(backbone).to(device)
    classifier = CognitiveStatusClassification(backbone).to(device)

    reg_criterion = nn.HuberLoss(delta=1.5)
    cls_criterion = nn.CrossEntropyLoss()

    params = [
        {'params': backbone.parameters()},
        {'params': regressor.net.parameters()},
        {'params': classifier.net.parameters()}
    ]

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )

    return regressor, classifier, reg_criterion, cls_criterion, optimizer, scheduler

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
def create_dataloader(X, y, z, batch_size=32, shuffle=True):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    z = torch.tensor(z, dtype=torch.long)

    dataset = TensorDataset(X, y, z)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

# ========== TRAINING & TESTING (INDIVIDUAL) ========== #

# train one batch
def train_reg_step(x, y, regressor, reg_criterion, reg_optimizer):
    x = x.to(device)
    y = y.to(device)

    reg_optimizer.zero_grad()
    pred = regressor(x)
    loss = reg_criterion(pred, y)
    loss.backward()

    # gradient clipping just to be safe
    torch.nn.utils.clip_grad_norm_(regressor.parameters(), max_norm=1.0)

    reg_optimizer.step()

    return loss.item()

# train an epoch
def train_reg_one_epoch(loader, regressor, reg_criterion, reg_optimizer):
    total = 0.0
    count = 0

    regressor.train()

    for batch_x, batch_y, _ in loader:
        loss = train_reg_step(batch_x, batch_y, regressor, reg_criterion, reg_optimizer)
        total += loss
        count += 1

    return total / count

# test with a test loader
def test_reg(loader, regressor, reg_criterion):
    regressor.eval()
    total_loss = 0.0
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_x, batch_y, _ in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            pred = regressor(batch_x)
            loss = reg_criterion(pred, batch_y)

            total_loss += loss.item()
            predictions.extend(pred.cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())

    mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
    rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals)) ** 2))
    return total_loss / len(loader), mae, rmse

# train one batch
def train_cls_step(x, z, classifier, cls_criterion, cls_optimizer):
    x = x.to(device)
    z = z.to(device)

    cls_optimizer.zero_grad()
    logits = classifier(x)
    loss = cls_criterion(logits, z)
    loss.backward()

    # gradient clipping just to be safe
    torch.nn.utils.clip_grad_norm_(classifier.net.parameters(), max_norm=1.0)

    cls_optimizer.step()

    return loss.item()

# train an epoch
def train_cls_one_epoch(loader, classifier, cls_criterion, cls_optimizer):
    total = 0.0
    count = 0

    classifier.train()

    for batch_x, _, batch_z in loader:
        loss = train_cls_step(batch_x, batch_z, classifier, cls_criterion, cls_optimizer)
        total += loss
        count += 1

    return total / count

# test with a test loader
def test_cls(loader, classifier, cls_criterion):
    classifier.eval()
    total_loss = 0.0
    z_true = []
    z_pred = []

    with torch.no_grad():
        for batch_x, _, batch_z in loader:
            batch_x = batch_x.to(device)
            batch_z = batch_z.to(device)

            logits = classifier(batch_x)
            loss = cls_criterion(logits, batch_z)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            z_true.extend(batch_z.cpu().tolist())
            z_pred.extend(preds.cpu().tolist())

    accuracy = accuracy_score(z_true, z_pred)
    f1 = f1_score(z_true, z_pred, average='macro')
    confusion = confusion_matrix(z_true, z_pred, labels=list(range(len(cog_statuses))))

    disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=list(range(len(cog_statuses))))
    disp.plot(cmap=plt.cm.Blues)  # Use Matplotlib's colormap
    plt.savefig('confusion_matrix.png')

    return total_loss / len(loader), accuracy, f1, confusion

# ========== TRAINING & TESTING (MULTITASK) ========== #

# train one batch
def train_mt_step(x, y, z, regressor, classifier, reg_criterion, cls_criterion, optimizer, lam=0.5):
    x = x.to(device)
    y = y.to(device)
    z = z.to(device)

    optimizer.zero_grad()

    pred_mmse = regressor(x)          # (B, 1)
    logits = classifier(x)           # (B, 3)

    loss_mmse = reg_criterion(pred_mmse, y)
    loss_cog = cls_criterion(logits, z)

    # custom loss: L_total = λL_mmse + (1-λ)L_cog
    loss = lam * loss_mmse + (1.0 - lam) * loss_cog
    loss.backward()

    params = []
    for group in optimizer.param_groups:
        params.extend(group['params'])
    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

    optimizer.step()

    return loss.item(), loss_mmse.item(), loss_cog.item()

# train an epoch
def train_mt_one_epoch(loader, regressor, classifier, reg_criterion, cls_criterion, optimizer, lam=0.5):
    regressor.train()
    classifier.train()

    total, total_mmse, total_cog = 0.0, 0.0, 0.0
    count = 0

    for batch_x, batch_y, batch_z in loader:
        loss, loss_mmse, loss_cog = train_mt_step(
            batch_x, batch_y, batch_z,
            regressor, classifier,
            reg_criterion, cls_criterion,
            optimizer,
            lam=lam
        )
        total += loss
        total_mmse += loss_mmse
        total_cog += loss_cog
        count += 1

    return total / count, total_mmse / count, total_cog / count

# ========== SAVE/LOAD WEIGHTS ========== #

# save weights and biases
def save(fp: Path, model):
    torch.save(model.state_dict(), str(fp))

# load weights and biases
def load(fp: Path, model):
    model.load_state_dict(torch.load(str(fp), map_location=device))
