# Main workflow (v3.0)

import json
from pathlib import Path

import numpy as np
import torch

from ml import model
from processing import pipeline
from utils import cache

# ansi color codes
RESET = '\033[0m'
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
CYAN = '\033[36m'
MAGENTA = '\033[35m'
BOLD = '\033[1m'

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False

TRAIN_JSON = Path('data_jsons/train.json')
VAL_JSON = Path('data_jsons/val.json')
TEST_JSON = Path('data_jsons/test.json')

REG_WEIGHTS_PATH = Path('models/model_weights_reg.pth')
CLS_WEIGHTS_PATH = Path('models/model_weights_cls.pth')
SCALER_PATH = Path('models/model_scaler.pkl')

FEATURE_DIR = Path('models/features')
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

# load up jsons
def load_split(json_path: Path):
    with json_path.open('r', encoding='utf-8') as f:
        payload = json.load(f)

    return payload['data']

# process split
def process_split(json_path: Path, split_name: str, use_cache=None, augment=True):
    if use_cache is None:
        use_cache = {
            'transcript': True,
            'acoustics': False,
            'linguistics': False,
            'semantics': True,
            'embeddings': True
        }

    print(f'\n{BOLD}{CYAN}Loading {split_name} split...{RESET}')

    data = load_split(json_path)

    print(f'{BLUE}   {split_name.title()} samples: {len(data)}{RESET}')
    print(f'\n{BOLD}{CYAN}Cleaning up {split_name} data...{RESET}')

    pipeline.clean_up_all(data)

    print(f'{BOLD}{GREEN}Done cleaning up {split_name} data ✓{RESET}')

    if augment:
        print(f'\n{BOLD}{CYAN}Augmenting {split_name} data...{RESET}')
        data = pipeline.augment_all(data)
        print(f'{BOLD}{GREEN}Done augmenting {split_name} data ✓{RESET}')

    print(f'\n{BOLD}{CYAN}Transcribing {split_name} data...{RESET}')
    pipeline.transcribe_all(data, use_cache_transcripts=use_cache['transcript'])
    print(f'{BOLD}{GREEN}Done transcribing {split_name} data ✓{RESET}')

    print(f'\n{BOLD}{CYAN}Extracting features from {split_name} data...{RESET}')
    pipeline.extract_features_all(data, use_cache_acoustics=use_cache['acoustics'], use_cache_linguistics=use_cache['linguistics'], use_cache_semantics=use_cache['semantics'])
    print(f'{BOLD}{GREEN}Done extracting features from {split_name} data ✓{RESET}')

    print(f'\n{BOLD}{CYAN}Generating embeddings from {split_name} data...{RESET}')
    pipeline.gen_embeddings_all(data, use_cache_embeddings=use_cache['embeddings'], batch_size=4)
    print(f'{BOLD}{GREEN}Done generating embeddings from {split_name} data ✓{RESET}')

    X = [d['features'] for d in data]
    y = [d['mmse'] for d in data]
    z = [model.cog_statuses.index(d['diagnosis']) for d in data]

    return X, y, z

# training with proper train/val/test split
def train(
    X_train_scaled, X_val_scaled, X_test_scaled,
    y_train, y_val, y_test,
    z_train, z_val, z_test,
    epochs_reg=40,
    epochs_cls=20
):
    # create dataloaders
    train_loader = model.create_dataloader(X_train_scaled, y_train, z_train, batch_size=64)
    val_loader = model.create_dataloader(X_val_scaled, y_val, z_val, batch_size=64)
    test_loader = model.create_dataloader(X_test_scaled, y_test, z_test, batch_size=64)

    REG_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # regression first
    print(f"\n{BOLD}{CYAN}Phase 1: Training regressor + backbone for {epochs_reg} epochs...{RESET}")

    best_val_mae = float('inf')
    best_reg_epoch = -1
    best_reg_snapshot = {
        'val_loss': None,
        'val_mae': None,
        'val_rmse': None
    }

    for epoch in range(epochs_reg):
        train_loss = model.train_reg_one_epoch(train_loader)
        val_loss, val_mae, val_rmse = model.test_reg(val_loader)

        # reduce learning rate if val loss plateaus
        model.scheduler.step(val_loss)

        # save model if validation MAE improves
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_reg_epoch = epoch + 1
            best_reg_snapshot = {
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_rmse': val_rmse
            }
            model.save_reg(REG_WEIGHTS_PATH)
            model.save_scaler(SCALER_PATH)
            indicator = " 💾"
        else:
            indicator = ""

        color = GREEN if indicator else BLUE
        print(
            f"{color}reg epoch {epoch + 1:02d}/{epochs_reg} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_mae: {val_mae:.4f} | "
            f"val_rmse: {val_rmse:.4f}{indicator}{RESET}"
        )

    print(f"\n{GREEN}🎉 Regression training complete! Best val MAE: {best_val_mae:.4f} (epoch {best_reg_epoch}){RESET}")

    # reload best regression model
    model.load_reg(REG_WEIGHTS_PATH)

    # classifier next
    print(f"\n{BOLD}{CYAN}Phase 2: Training classifier head for {epochs_cls} epochs...{RESET}")

    best_val_f1 = -1.0
    best_cls_epoch = -1
    best_cls_snapshot = {
        'val_loss': None,
        'val_acc': None,
        'val_f1': None,
        'val_cm': None
    }

    for epoch in range(epochs_cls):
        # train classification for one epoch
        model.classifier.train()
        cls_losses = []
        for batch_x, _, batch_z in train_loader:
            cls_loss = model.train_cls_step(batch_x, batch_z)
            cls_losses.append(cls_loss)

        train_cls_loss = float(np.mean(cls_losses)) if cls_losses else float('nan')

        # validate classification
        val_cls_loss, val_acc, val_f1, val_cm = model.test_cls(val_loader)

        # save best by macro-F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_cls_epoch = epoch + 1
            best_cls_snapshot = {
                'val_loss': val_cls_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_cm': val_cm
            }
            model.save_cls(CLS_WEIGHTS_PATH)
            indicator = " 💾"
        else:
            indicator = ""

        color = GREEN if indicator else BLUE
        print(
            f"{color}cls epoch {epoch + 1:02d}/{epochs_cls} | "
            f"train_loss: {train_cls_loss:.4f} | "
            f"val_loss: {val_cls_loss:.4f} | "
            f"val_acc: {val_acc:.4f} | "
            f"val_f1(macro): {val_f1:.4f}{indicator}{RESET}"
        )

    print(f"\n{GREEN}🎉 Classification training complete! Best val macro-F1: {best_val_f1:.4f} (epoch {best_cls_epoch}){RESET}")

    # final test eval
    print(f"\n{BOLD}{MAGENTA}FINAL TEST SET REPORT {RESET}")

    # Regression test (load best regressor)
    model.load_reg(REG_WEIGHTS_PATH)
    test_loss_reg, test_mae, test_rmse = model.test_reg(test_loader)

    # Classification test (load best classifier)
    model.load_cls(CLS_WEIGHTS_PATH)
    test_loss_cls, test_acc, test_f1, test_cm = model.test_cls(test_loader)

    # Print everything at once
    print(f"\n{BOLD}{MAGENTA}Regression (MMSE){RESET}")
    print(f"{MAGENTA}  test_loss: {test_loss_reg:.4f}{RESET}")
    print(f"{MAGENTA}  test_mae:  {test_mae:.4f}{RESET}")
    print(f"{MAGENTA}  test_rmse: {test_rmse:.4f}{RESET}")

    print(f"\n{BOLD}{MAGENTA}Classification (Cognitive Status){RESET}")
    print(f"{MAGENTA}  test_loss:      {test_loss_cls:.4f}{RESET}")
    print(f"{MAGENTA}  test_acc:       {test_acc:.4f}{RESET}")
    print(f"{MAGENTA}  test_f1(macro): {test_f1:.4f}{RESET}")
    print(f"{MAGENTA}  test_confusion_matrix (rows=true, cols=pred):{RESET}")

    labels = model.cog_statuses
    header = "".ljust(12) + "".join(lbl.rjust(8) for lbl in labels)
    print(f"{MAGENTA}{header}{RESET}")

    for i, lbl in enumerate(labels):
        row = lbl.ljust(12) + "".join(f"{test_cm[i, j]:8d}" for j in range(len(labels)))
        print(f"{MAGENTA}{row}{RESET}")

# main flow
def main():
    use_cache_list = {
        'transcript': cache.ask('transcripts'),
        'acoustics': cache.ask('acoustic features'),
        'linguistics': cache.ask('linguistic features'),
        'semantics': cache.ask('LLM-generated semantic features'),
        'embeddings': cache.ask('audio embeddings')
    }

    X_train, y_train, z_train = process_split(TRAIN_JSON, 'train', use_cache=use_cache_list, augment=True)
    X_val, y_val, z_val = process_split(VAL_JSON, 'validation', use_cache=use_cache_list, augment=False)
    X_test, y_test, z_test = process_split(TEST_JSON, 'test', use_cache=use_cache_list, augment=False)
    print(f"\n{GREEN}Done processing all data!{RESET}")

    # scale features (fit on train only)
    print(f"\n{BOLD}{CYAN}Fitting scaler on train data...{RESET}")
    model.fit_scaler(X_train)

    X_train_scaled = model.transform_features(X_train)
    X_val_scaled = model.transform_features(X_val)
    X_test_scaled = model.transform_features(X_test)

    # save scaled features + labels
    np.save(FEATURE_DIR / 'X_train_scaled.npy', X_train_scaled)
    np.save(FEATURE_DIR / 'X_val_scaled.npy', X_val_scaled)
    np.save(FEATURE_DIR / 'X_test_scaled.npy', X_test_scaled)
    np.save(FEATURE_DIR / 'y_train.npy', y_train)
    np.save(FEATURE_DIR / 'y_val.npy', y_val)
    np.save(FEATURE_DIR / 'y_test.npy', y_test)
    np.save(FEATURE_DIR / 'z_train.npy', z_train)
    np.save(FEATURE_DIR / 'z_val.npy', z_val)
    np.save(FEATURE_DIR / 'z_test.npy', z_test)

    print(f"\n{GREEN}✓ Saved scaled features + labels to {FEATURE_DIR}{RESET}")

    train(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, z_train, z_val, z_test)

if __name__ == '__main__':
    main()
