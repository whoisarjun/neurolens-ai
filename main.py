# Main workflow (v2.10)

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

MODEL_WEIGHTS_PATH = Path('models/model_weights.pth')
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
    pipeline.extract_features_all(data, use_cache_semantics=use_cache['semantics'])
    print(f'{BOLD}{GREEN}Done extracting features from {split_name} data ✓{RESET}')

    print(f'\n{BOLD}{CYAN}Generating embeddings from {split_name} data...{RESET}')
    pipeline.gen_embeddings_all(data, use_cache_embeddings=use_cache['embeddings'])
    print(f'{BOLD}{GREEN}Done generating embeddings from {split_name} data ✓{RESET}')

    X = [d['features'] for d in data]
    y = [d['mmse'] for d in data]

    return X, y

# training with proper train/val/test split
def train(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, epochs=40):
    # create dataloaders
    train_loader = model.create_dataloader(X_train_scaled, y_train, batch_size=64)
    val_loader = model.create_dataloader(X_val_scaled, y_val, batch_size=64)
    test_loader = model.create_dataloader(X_test_scaled, y_test, batch_size=64)

    MODEL_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{BOLD}{CYAN}Training neural network for {epochs} epochs...{RESET}")
    print(f"{BLUE}Using validation set for early stopping, test set held out until end{RESET}")

    best_val_mae = float('inf')

    for epoch in range(epochs):
        train_loss = model.train_one_epoch(train_loader)
        val_loss, val_mae, val_rmse = model.test(val_loader)

        # reduce learning rate if val mae plateaus
        model.scheduler.step(val_loss)

        # save model if validation MAE improves
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            model.save(MODEL_WEIGHTS_PATH)
            model.save_scaler(SCALER_PATH)
            indicator = " 💾"  # indicate save best
        else:
            indicator = ""

        color = GREEN if indicator else BLUE
        print(
            f"{color}epoch {epoch + 1:02d}/{epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_mae: {val_mae:.4f} | "
            f"val_rmse: {val_rmse:.4f}{indicator}{RESET}"
        )

    print(f"\n{GREEN}🎉 Training complete! Best validation MAE: {best_val_mae:.4f}{RESET}")

    # Final evaluation on held-out test set
    print(f"\n{BOLD}{MAGENTA}Evaluating on held-out test set...{RESET}")
    model.load(MODEL_WEIGHTS_PATH)  # Load best model
    test_loss, test_mae, test_rmse = model.test(test_loader)
    print(f"{MAGENTA}Final Test Loss: {test_loss:.4f}{RESET}")
    print(f"{BOLD}{MAGENTA}Final Test MAE: {test_mae:.4f}{RESET}")
    print(f"{BOLD}{MAGENTA}Final Test RMSE: {test_rmse:.4f}{RESET}")

# main flow
def main():
    use_cache_list = {
        'transcript': cache.ask('transcripts'),
        'semantics': cache.ask('LLM-generated semantic features'),
        'embeddings': cache.ask('audio embeddings')
    }

    X_train, y_train = process_split(TRAIN_JSON, 'train', use_cache=use_cache_list, augment=True)
    X_val, y_val = process_split(VAL_JSON, 'validation', use_cache=use_cache_list, augment=False)
    X_test, y_test = process_split(TEST_JSON, 'test', use_cache=use_cache_list, augment=False)
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

    print(f"\n{GREEN}✓ Saved scaled features + labels to {FEATURE_DIR}{RESET}")

    train(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test)

if __name__ == '__main__':
    main()
