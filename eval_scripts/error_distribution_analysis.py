# Error distribution analysis script

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

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

    dataset_data = defaultdict(lambda: {'true': [], 'pred': []})
    for i, dataset in enumerate(dataset_names):
        dataset_data[dataset]['true'].append(y_test[i])
        dataset_data[dataset]['pred'].append(predictions[i])

    for dataset in dataset_data:
        dataset_data[dataset]['true'] = np.array(dataset_data[dataset]['true'])
        dataset_data[dataset]['pred'] = np.array(dataset_data[dataset]['pred'])

    # ========== MAE Distribution by Dataset ========== #
    print("\nGenerating MAE distribution plot...")

    plt.figure(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(dataset_data)))

    for i, (dataset, color) in enumerate(zip(sorted(dataset_data.keys()), colors)):
        true_vals = dataset_data[dataset]['true']
        pred_vals = dataset_data[dataset]['pred']

        absolute_errors = np.abs(true_vals - pred_vals)

        # density plot with KDE
        if len(absolute_errors) > 1:
            density = stats.gaussian_kde(absolute_errors)
            x_range = np.linspace(0, max(absolute_errors) * 1.1, 200)
            plt.plot(x_range, density(x_range), label=dataset, color=color, linewidth=2)

    plt.xlabel('|MMSE_true - MMSE_pred|', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('MAE Distribution by Dataset', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / 'mae_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved to {EVAL_DIR / 'mae_distribution.png'}")

    # ========== Cumulative Distribution Functions ========== #
    print("Generating CDF plot...")

    plt.figure(figsize=(10, 6))

    for i, (dataset, color) in enumerate(zip(sorted(dataset_data.keys()), colors)):
        true_vals = dataset_data[dataset]['true']
        pred_vals = dataset_data[dataset]['pred']

        absolute_errors = np.abs(true_vals - pred_vals)

        # sort errors and calc cumulative percentages
        sorted_errors = np.sort(absolute_errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100

        plt.plot(sorted_errors, cumulative, label=dataset, color=color, linewidth=2)

    plt.xlabel('MAE Threshold', fontsize=12)
    plt.ylabel('% of Samples ≤ MAE Threshold', fontsize=12)
    plt.title('Cumulative Distribution of Absolute Error', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / 'cdf_absolute_error.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved to {EVAL_DIR / 'cdf_absolute_error.png'}")

    # ========== Bland-Altman Plot ========== #
    print("Generating Bland-Altman plot...")

    plt.figure(figsize=(12, 8))

    for i, (dataset, color) in enumerate(zip(sorted(dataset_data.keys()), colors)):
        true_vals = dataset_data[dataset]['true']
        pred_vals = dataset_data[dataset]['pred']

        means = (true_vals + pred_vals) / 2
        diffs = pred_vals - true_vals

        plt.scatter(means, diffs, label=dataset, color=color, alpha=0.6, s=50)

    # calc mean diff and lims of agreement
    all_true = np.concatenate([dataset_data[d]['true'] for d in dataset_data.keys()])
    all_pred = np.concatenate([dataset_data[d]['pred'] for d in dataset_data.keys()])
    all_means = (all_true + all_pred) / 2
    all_diffs = all_pred - all_true

    mean_diff = np.mean(all_diffs)
    std_diff = np.std(all_diffs)

    # plot mean and lims of agreement
    plt.axhline(mean_diff, color='black', linestyle='--', linewidth=2, label=f'Mean Diff: {mean_diff:.2f}')
    plt.axhline(mean_diff + 1.96 * std_diff, color='red', linestyle='--', linewidth=2,
                label=f'+1.96 SD: {mean_diff + 1.96 * std_diff:.2f}')
    plt.axhline(mean_diff - 1.96 * std_diff, color='red', linestyle='--', linewidth=2,
                label=f'-1.96 SD: {mean_diff - 1.96 * std_diff:.2f}')

    plt.xlabel('Mean(MMSE_true, MMSE_pred)', fontsize=12)
    plt.ylabel('MMSE_pred - MMSE_true', fontsize=12)
    plt.title('Bland-Altman Plot', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / 'bland_altman_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved to {EVAL_DIR / 'bland_altman_analysis.png'}")

    # ========== True MMSE vs Predicted MMSE ========== #
    print("Generating True MMSE vs Predicted MMSE plot...")

    plt.figure(figsize=(12, 8))

    # scatter plot with color-coded datasets
    for i, (dataset, color) in enumerate(zip(sorted(dataset_data.keys()), colors)):
        true_vals = dataset_data[dataset]['true']
        pred_vals = dataset_data[dataset]['pred']

        plt.scatter(true_vals, pred_vals, label=dataset, color=color, alpha=0.5, s=50)

    all_true = np.concatenate([dataset_data[d]['true'] for d in dataset_data.keys()])
    all_pred = np.concatenate([dataset_data[d]['pred'] for d in dataset_data.keys()])

    # perfect pred line
    min_val = min(all_true.min(), all_pred.min())
    max_val = max(all_true.max(), all_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2,
             label='Perfect Prediction (y=x)', zorder=100)

    # calc and plot binned avg preds
    sort_idx = np.argsort(all_true)
    sorted_true = all_true[sort_idx]
    sorted_pred = all_pred[sort_idx]

    n_bins = 15
    bin_edges = np.linspace(sorted_true.min(), sorted_true.max(), n_bins + 1)
    bin_centers = []
    bin_mean_preds = []

    for i in range(n_bins):
        mask = (sorted_true >= bin_edges[i]) & (sorted_true < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_mean_preds.append(sorted_pred[mask].mean())

    # plot avg pred curve
    plt.plot(bin_centers, bin_mean_preds, color='red', linewidth=3, linestyle='-',
             marker='o', markersize=8, label='Average Prediction', zorder=101)

    plt.xlabel('True MMSE Score', fontsize=12)
    plt.ylabel('Predicted MMSE Score', fontsize=12)
    plt.title('True MMSE vs Predicted MMSE', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xlim([min_val - 1, max_val + 1])
    plt.ylim([min_val - 1, max_val + 1])
    plt.tight_layout()
    plt.savefig(EVAL_DIR / 'true_vs_predicted_mmse.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved to {EVAL_DIR / 'true_vs_predicted_mmse.png'}")

    print("\n" + "=" * 60)
    print("Error distribution analysis complete!")
    print("=" * 60)

if __name__ == '__main__':
    main()