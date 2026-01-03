# Best and worst prediction analysis script

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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

    # mae calc and sort
    absolute_errors = np.abs(y_test - predictions)

    results_df = pd.DataFrame({
        'True_MMSE': y_test,
        'Predicted_MMSE': predictions,
        'Dataset': dataset_names,
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