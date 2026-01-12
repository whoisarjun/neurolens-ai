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
    regressor, classifier,
    reg_criterion, cls_criterion,
    optimizer, scheduler,
    lam=0.5,
    epochs=50,
    verbose=True
):
    train_loader = model.create_dataloader(X_train_scaled, y_train, z_train, batch_size=64)
    val_loader = model.create_dataloader(X_val_scaled, y_val, z_val, batch_size=64)
    test_loader = model.create_dataloader(X_test_scaled, y_test, z_test, batch_size=64)

    best_score = -float('inf')
    best_epoch = -1

    for epoch in range(epochs):
        train_loss, train_mmse, train_cog = model.train_mt_one_epoch(
            train_loader,
            regressor, classifier,
            reg_criterion, cls_criterion,
            optimizer,
            lam=lam
        )

        val_reg_loss, val_mae, val_rmse = model.test_reg(val_loader, regressor, reg_criterion)
        val_cls_loss, val_acc, val_f1, _ = model.test_cls(val_loader, classifier, cls_criterion)

        scheduler.step(val_reg_loss)

        # balanced model selection score: prefer low MAE + decent macro-F1
        alpha = 2
        score = (-val_mae) + alpha * val_f1

        if score > best_score:
            best_score = score
            best_epoch = epoch + 1

            # Save both heads (they share backbone weights anyway)
            model.save(REG_WEIGHTS_PATH, regressor)
            model.save(CLS_WEIGHTS_PATH, classifier)
            indicator = " 💾"
        else:
            indicator = ""

        if verbose:
            color = GREEN if indicator else BLUE
            print(
                f"{color}mt epoch {epoch+1:02d}/{epochs} | "
                f"train_total: {train_loss:.4f} (mmse {train_mmse:.4f}, cog {train_cog:.4f}) | "
                f"val_mae: {val_mae:.3f} | val_rmse: {val_rmse:.3f} | "
                f"val_f1: {val_f1:.3f} | val_acc: {val_acc:.3f}{indicator}{RESET}"
            )

    if verbose:
        print(f"\n{GREEN}🎉 Multitask training done. Best score at epoch {best_epoch}{RESET}")

    # final test eval
    model.load(REG_WEIGHTS_PATH, regressor)
    model.load(CLS_WEIGHTS_PATH, classifier)

    test_reg_loss, test_mae, test_rmse = model.test_reg(test_loader, regressor, reg_criterion)
    test_cls_loss, test_acc, test_f1, test_cm = model.test_cls(test_loader, classifier, cls_criterion)

    return test_mae, test_rmse, test_acc, test_f1, test_cm

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
    model.save_scaler(SCALER_PATH)

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

    backbone = model.new_backbone()
    regressor, classifier, reg_criterion, cls_criterion, optimizer, scheduler = model.new_multitask(backbone)

    test_mae, test_rmse, test_acc, test_f1, test_cm = train(
        X_train_scaled, X_val_scaled, X_test_scaled,
        y_train, y_val, y_test,
        z_train, z_val, z_test,
        regressor, classifier,
        reg_criterion, cls_criterion,
        optimizer, scheduler,
        lam=0.45,
        epochs=50
    )

    print(f"\n{BOLD}{MAGENTA}{'=' * 60}{RESET}")
    print(f"{BOLD}{MAGENTA}FINAL TEST SET RESULTS{RESET}")
    print(f"{BOLD}{MAGENTA}{'=' * 60}{RESET}")
    print(f"{YELLOW}Regression → MAE: {test_mae:.3f} | RMSE: {test_rmse:.3f}{RESET}")
    print(f"{YELLOW}Classification → Accuracy: {test_acc:.3f} | F1: {test_f1:.3f}{RESET}")
    print(f"{CYAN}Confusion Matrix:\n{test_cm}{RESET}")
    print(f"{BOLD}{MAGENTA}{'=' * 60}{RESET}\n")

if __name__ == '__main__':
    main()
