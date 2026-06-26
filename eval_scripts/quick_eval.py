# Quick evaluation script

import sys
from pathlib import Path

# allow `python eval_scripts/<name>.py` from the repo root to import project packages
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from eval_scripts.eval_stats import bootstrap_classification, bootstrap_regression, format_mean_std
from ml import model

# bootstrap resamples used to estimate the mean ± std of each test metric
N_BOOTSTRAP = 1000

REG_WEIGHTS_PATH = Path('models/model_weights_reg.pth')
CLS_WEIGHTS_PATH = Path('models/model_weights_cls.pth')
SCALER_PATH = Path('models/model_scaler.pkl')
FEATURE_DIR = Path('models/features')
EVAL_DIR = Path('eval_results')
EVAL_DIR.mkdir(parents=True, exist_ok=True)

def require_files(paths):
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        print('\nMissing required files. Train a current model with `python main.py` first:')
        for entry in missing:
            print(f'  - {entry}')
        sys.exit(1)


def load_weights(*pairs):
    try:
        for path, net in pairs:
            model.load(path, net)
    except RuntimeError as exc:
        print(
            '\nFailed to load weights into the current 227-dim architecture. '
            'The tracked weights may belong to an older model; retrain with `python main.py`.\n'
        )
        print(exc)
        sys.exit(1)


def main():
    print("Loading test data...")

    require_files([
        FEATURE_DIR / 'X_test_scaled.npy', FEATURE_DIR / 'y_test.npy',
        FEATURE_DIR / 'z_test.npy', FEATURE_DIR / 'y_test_mask.npy',
        FEATURE_DIR / 'z_test_mask.npy',
        REG_WEIGHTS_PATH, CLS_WEIGHTS_PATH, SCALER_PATH,
    ])

    # load scaled features and labels
    X_test_scaled = np.load(FEATURE_DIR / 'X_test_scaled.npy')
    y_test = np.load(FEATURE_DIR / 'y_test.npy')
    z_test = np.load(FEATURE_DIR / 'z_test.npy')
    y_test_mask = np.load(FEATURE_DIR / 'y_test_mask.npy').astype(bool)
    z_test_mask = np.load(FEATURE_DIR / 'z_test_mask.npy').astype(bool)

    print(f"Test samples: {len(X_test_scaled)}")

    model.load_scaler(SCALER_PATH)

    # create models
    backbone = model.new_backbone()
    regressor = model.MMSERegression(backbone).to(model.device)
    classifier = model.CognitiveStatusClassification(backbone).to(model.device)

    # load weights
    load_weights((REG_WEIGHTS_PATH, regressor), (CLS_WEIGHTS_PATH, classifier))

    # inference is deterministic, so run a single forward pass and bootstrap the
    # test set to obtain a mean ± std for each metric.
    regressor.eval()
    classifier.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(model.device)
        reg_preds = regressor(X_tensor).squeeze(-1).cpu().numpy()
        cls_preds = torch.argmax(classifier(X_tensor), dim=1).cpu().numpy()

    reg_stats = bootstrap_regression(reg_preds[y_test_mask], y_test[y_test_mask], N_BOOTSTRAP)
    cls_stats = bootstrap_classification(z_test[z_test_mask], cls_preds[z_test_mask], N_BOOTSTRAP)

    # prints
    print("\n" + "=" * 60)
    print(f"QUICK EVALUATION RESULTS (bootstrap mean ± std, {N_BOOTSTRAP} resamples)")
    print("=" * 60)
    print(
        f"Regression → MAE: {format_mean_std(*reg_stats['MAE'])} | "
        f"RMSE: {format_mean_std(*reg_stats['RMSE'])} | R²: {format_mean_std(*reg_stats['R²'])}"
    )
    print(
        f"Classification → Accuracy: {format_mean_std(*cls_stats['Accuracy'])} | "
        f"Macro-F1: {format_mean_std(*cls_stats['Macro-F1'])}"
    )
    print("=" * 60)

    # confusion matrix (point estimate over the full test split, not resampled)
    if z_test_mask.any():
        confusion = confusion_matrix(
            z_test[z_test_mask], cls_preds[z_test_mask],
            labels=list(range(len(model.cog_statuses)))
        )
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion,
            display_labels=model.cog_statuses
        )
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(EVAL_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        print(f"\nConfusion matrix saved to {EVAL_DIR / 'confusion_matrix.png'}")
        plt.close()
    else:
        print("\nConfusion matrix skipped: no diagnosis labels in test split.")

if __name__ == '__main__':
    main()
