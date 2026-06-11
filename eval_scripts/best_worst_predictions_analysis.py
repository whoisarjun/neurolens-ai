# Best and worst prediction analysis script

import json
import sys
from pathlib import Path

# allow `python eval_scripts/<name>.py` from the repo root to import project packages
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from ml import model


def require_files(paths):
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        print('\nMissing required files. Train a current model with `python main.py` first:')
        for entry in missing:
            print(f'  - {entry}')
        sys.exit(1)


def load_weights(path, net):
    try:
        model.load(path, net)
    except RuntimeError as exc:
        print(
            '\nFailed to load weights into the current 227-dim architecture. '
            'The tracked weights may belong to an older model; retrain with `python main.py`.\n'
        )
        print(exc)
        sys.exit(1)

TEST_JSON = Path('data_jsons/test.json')
REG_WEIGHTS_PATH = Path('models/model_weights_reg.pth')
SCALER_PATH = Path('models/model_scaler.pkl')
FEATURE_DIR = Path('models/features')
EVAL_DIR = Path('eval_results')
EVAL_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Loading test data...")

    require_files([
        TEST_JSON,
        FEATURE_DIR / 'X_test_scaled.npy', FEATURE_DIR / 'y_test.npy',
        REG_WEIGHTS_PATH, SCALER_PATH,
    ])

    with TEST_JSON.open('r', encoding='utf-8') as f:
        test_data = json.load(f)['data']

    # dataset names
    dataset_names = []
    for entry in test_data:
        output_path = entry['output']
        parts = output_path.split('/')
        dataset = parts[1] if len(parts) > 1 else parts[0]
        dataset_names.append(dataset)

    # load everything
    X_test_scaled = np.load(FEATURE_DIR / 'X_test_scaled.npy')
    y_test = np.load(FEATURE_DIR / 'y_test.npy')

    model.load_scaler(SCALER_PATH)
    backbone = model.new_backbone()
    regressor = model.MMSERegression(backbone).to(model.device)
    load_weights(REG_WEIGHTS_PATH, regressor)

    # predict
    regressor.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(model.device)
        predictions = regressor(X_tensor).cpu().numpy().flatten()

    # keep only samples with a real MMSE label (unlabeled entries are NaN)
    dataset_names = np.array(dataset_names)
    valid_mask = np.isfinite(y_test) & np.isfinite(predictions)
    y_valid = y_test[valid_mask]
    pred_valid = predictions[valid_mask]
    dataset_valid = dataset_names[valid_mask]

    # mae calc and sort
    absolute_errors = np.abs(y_valid - pred_valid)

    results_df = pd.DataFrame({
        'True_MMSE': y_valid,
        'Predicted_MMSE': pred_valid,
        'Dataset': dataset_valid,
        'MAE': absolute_errors
    })

    best_predictions = results_df.nsmallest(10, 'MAE')
    worst_predictions = results_df.nlargest(10, 'MAE')

    # print and save
    best_predictions = best_predictions[['True_MMSE', 'Predicted_MMSE', 'Dataset', 'MAE']]
    worst_predictions = worst_predictions[['True_MMSE', 'Predicted_MMSE', 'Dataset', 'MAE']]
    best_predictions.to_csv(EVAL_DIR / 'best_predictions.csv', index=False)
    worst_predictions.to_csv(EVAL_DIR / 'worst_predictions.csv', index=False)

    print("\n" + "=" * 60)
    print("TOP 10 BEST PREDICTIONS")
    print("=" * 60)
    print(best_predictions.to_string(index=False))

    print("\n" + "=" * 60)
    print("TOP 10 WORST PREDICTIONS")
    print("=" * 60)
    print(worst_predictions.to_string(index=False))

    print(f"\n{'=' * 60}")
    print(f"Best predictions saved to {EVAL_DIR / 'best_predictions.csv'}")
    print(f"Worst predictions saved to {EVAL_DIR / 'worst_predictions.csv'}")
    print("=" * 60)


if __name__ == '__main__':
    main()