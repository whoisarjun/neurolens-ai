# Dataset breakdown analysis script

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

from ml import model

TEST_JSON = Path('data_jsons/test.json')
REG_WEIGHTS_PATH = Path('models/model_weights_reg.pth')
SCALER_PATH = Path('models/model_scaler.pkl')
FEATURE_DIR = Path('models/features')
EVAL_DIR = Path('eval_results')
EVAL_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Loading test data...")

    with TEST_JSON.open('r', encoding='utf-8') as f:
        test_data = json.load(f)['data']

    # dataset names
    dataset_names = []
    for entry in test_data:
        output_path = entry['output']
        dataset = output_path.split('/')[1]
        dataset_names.append(dataset)

    # load everything
    X_test_scaled = np.load(FEATURE_DIR / 'X_test_scaled.npy')
    y_test = np.load(FEATURE_DIR / 'y_test.npy')

    model.load_scaler(SCALER_PATH)
    backbone = model.new_backbone()
    regressor = model.MMSERegression(backbone).to(model.device)
    model.load(REG_WEIGHTS_PATH, regressor)

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
        mae = np.mean(np.abs(true_vals - pred_vals))
        rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))
        r2 = r2_score(true_vals, pred_vals)

        results.append({
            'Dataset': dataset,
            'Samples': n_samples,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2
        })

        print(f"{dataset}: n={n_samples}, MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")

    # save results
    df = pd.DataFrame(results)
    df.to_csv(EVAL_DIR / 'dataset_breakdown.csv', index=False)

    print(f"\n{'=' * 60}")
    print(f"Results saved to {EVAL_DIR / 'dataset_breakdown.csv'}")

if __name__ == '__main__':
    main()