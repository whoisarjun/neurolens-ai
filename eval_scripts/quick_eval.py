# Quick evaluation script

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, r2_score

from ml import model

REG_WEIGHTS_PATH = Path('models/model_weights_reg.pth')
CLS_WEIGHTS_PATH = Path('models/model_weights_cls.pth')
SCALER_PATH = Path('models/model_scaler.pkl')
FEATURE_DIR = Path('models/features')
EVAL_DIR = Path('eval_results')
EVAL_DIR.mkdir(parents=True, exist_ok=True)

def compute_score(values):
    if not values['preds']:
        return None
    return r2_score(np.concatenate(values['targets']), np.concatenate(values['preds']))

def main():
    print("Loading test data...")

    # load scaled features and labels
    X_test_scaled = np.load(FEATURE_DIR / 'X_test_scaled.npy')
    y_test = np.load(FEATURE_DIR / 'y_test.npy')
    z_test = np.load(FEATURE_DIR / 'z_test.npy')
    y_test_mask = np.load(FEATURE_DIR / 'y_test_mask.npy')
    z_test_mask = np.load(FEATURE_DIR / 'z_test_mask.npy')

    print(f"Test samples: {len(X_test_scaled)}")

    model.load_scaler(SCALER_PATH)

    # create models
    backbone = model.new_backbone()
    regressor = model.MMSERegression(backbone).to(model.device)
    classifier = model.CognitiveStatusClassification(backbone).to(model.device)

    # load weights
    model.load(REG_WEIGHTS_PATH, regressor)
    model.load(CLS_WEIGHTS_PATH, classifier)

    # test loader
    test_loader = model.create_dataloader(
        X_test_scaled, y_test, z_test, y_test_mask, z_test_mask, batch_size=64, shuffle=False
    )

    # regression eval
    reg_criterion = torch.nn.HuberLoss(delta=1.5)
    _, mae, rmse = model.test_reg(test_loader, regressor, reg_criterion)

    # compute R^2
    regressor.eval()
    reg_values = {'preds': [], 'targets': []}
    with torch.no_grad():
        for xb, yb, _, y_mask, _ in test_loader:
            if not y_mask.any():
                continue
            xb = xb[y_mask]
            yb = yb[y_mask]
            xb = xb.to(model.device)
            yb = yb.to(model.device)
            preds = regressor(xb).squeeze()
            reg_values['preds'].append(preds.cpu().numpy())
            reg_values['targets'].append(yb.cpu().numpy())

    r2 = compute_score(reg_values)

    # classification eval
    cls_criterion = torch.nn.CrossEntropyLoss()
    _, accuracy, f1, confusion = model.test_cls(test_loader, classifier, cls_criterion)

    # prints
    print("\n" + "=" * 60)
    print("QUICK EVALUATION RESULTS")
    print("=" * 60)
    print(
        f"Regression → MAE: {model.format_metric(mae)} | "
        f"RMSE: {model.format_metric(rmse)} | R²: {model.format_metric(r2)}"
    )
    print(f"Classification → Accuracy: {model.format_metric(accuracy)} | Macro-F1: {model.format_metric(f1)}")
    print("=" * 60)

    # confusion matrix
    if confusion is not None:
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
