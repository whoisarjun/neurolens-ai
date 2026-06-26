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

from eval_scripts.eval_stats import bootstrap_regression, format_mean_std
from ml import model

# bootstrap resamples used to estimate the mean ± std of each metric
N_BOOTSTRAP = 1000


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

    # calc metrics for each dataset (bootstrap the valid samples for mean ± std)
    results = []

    print("\n" + "=" * 60)
    print(f"DATASET BREAKDOWN ANALYSIS (bootstrap mean ± std, {N_BOOTSTRAP} resamples)")
    print("=" * 60)

    for dataset in sorted(dataset_data.keys()):
        true_vals = np.array(dataset_data[dataset]['true'])
        pred_vals = np.array(dataset_data[dataset]['pred'])

        n_samples = len(true_vals)
        valid_mask = np.isfinite(true_vals) & np.isfinite(pred_vals)
        n_valid = int(valid_mask.sum())
        n_invalid = n_samples - n_valid

        # bootstrap_regression takes (preds, targets); R² is direction-sensitive
        stats = bootstrap_regression(pred_vals[valid_mask], true_vals[valid_mask], N_BOOTSTRAP)
        mae_mean, mae_std = stats['MAE']
        rmse_mean, rmse_std = stats['RMSE']
        r2_mean, r2_std = stats['R²']

        results.append({
            'Dataset': dataset,
            'Samples': n_samples,
            'Valid Samples': n_valid,
            'Invalid Samples': n_invalid,
            'MAE_mean': mae_mean, 'MAE_std': mae_std,
            'RMSE_mean': rmse_mean, 'RMSE_std': rmse_std,
            'R²_mean': r2_mean, 'R²_std': r2_std,
        })

        invalid_note = f", invalid={n_invalid}" if n_invalid else ""
        print(
            f"{dataset}: n={n_samples}, valid={n_valid}{invalid_note}, "
            f"MAE={format_mean_std(mae_mean, mae_std)}, "
            f"RMSE={format_mean_std(rmse_mean, rmse_std)}, "
            f"R²={format_mean_std(r2_mean, r2_std)}"
        )

    # save results
    df = pd.DataFrame(results)
    df.to_csv(EVAL_DIR / 'dataset_breakdown.csv', index=False)

    print(f"\n{'=' * 60}")
    print(f"Results saved to {EVAL_DIR / 'dataset_breakdown.csv'}")

if __name__ == '__main__':
    main()