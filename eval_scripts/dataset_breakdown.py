# Dataset breakdown analysis script

import json
import sys
from collections import defaultdict
from pathlib import Path

# allow `python eval_scripts/<name>.py` from the repo root to import project packages
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

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

    # group by dataset
    dataset_data = defaultdict(lambda: {'true': [], 'pred': []})
    for i, dataset in enumerate(dataset_names):
        dataset_data[dataset]['true'].append(y_test[i])
        dataset_data[dataset]['pred'].append(predictions[i])

    # calc metrics for each dataset
    results = []

    print("\n" + "=" * 60)
    print("DATASET BREAKDOWN ANALYSIS")
    print("=" * 60)

    for dataset in sorted(dataset_data.keys()):
        true_vals = np.array(dataset_data[dataset]['true'])
        pred_vals = np.array(dataset_data[dataset]['pred'])

        n_samples = len(true_vals)
        valid_mask = np.isfinite(true_vals) & np.isfinite(pred_vals)
        n_valid = int(valid_mask.sum())
        n_invalid = n_samples - n_valid

        if n_valid == 0:
            mae = np.nan
            rmse = np.nan
            r2 = np.nan
        else:
            true_valid = true_vals[valid_mask]
            pred_valid = pred_vals[valid_mask]

            mae = np.mean(np.abs(true_valid - pred_valid))
            rmse = np.sqrt(np.mean((true_valid - pred_valid) ** 2))

            if n_valid >= 2 and np.var(true_valid) > 0:
                r2 = r2_score(true_valid, pred_valid)
            else:
                r2 = np.nan

        results.append({
            'Dataset': dataset,
            'Samples': n_samples,
            'Valid Samples': n_valid,
            'Invalid Samples': n_invalid,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2
        })

        invalid_note = f", invalid={n_invalid}" if n_invalid else ""
        r2_display = f"{r2:.3f}" if np.isfinite(r2) else "N/A"
        print(
            f"{dataset}: n={n_samples}, valid={n_valid}{invalid_note}, "
            f"MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2_display}"
        )

    # save results
    df = pd.DataFrame(results)
    df.to_csv(EVAL_DIR / 'dataset_breakdown.csv', index=False)

    print(f"\n{'=' * 60}")
    print(f"Results saved to {EVAL_DIR / 'dataset_breakdown.csv'}")

if __name__ == '__main__':
    main()