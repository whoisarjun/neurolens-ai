# Quick evaluation script

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, r2_score

from ml import model

TEST_JSON = Path('data_jsons/test.json')
REG_WEIGHTS_PATH = Path('models/model_weights_reg.pth')
CLS_WEIGHTS_PATH = Path('models/model_weights_cls.pth')
SCALER_PATH = Path('models/model_scaler.pkl')
FEATURE_DIR = Path('models/features')
EVAL_DIR = Path('eval_results')
EVAL_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Loading test data...")

    # load scaled features and labels
    X_test_scaled = np.load(FEATURE_DIR / 'X_test_scaled.npy')
    y_test = np.load(FEATURE_DIR / 'y_test.npy')
    z_test = np.load(FEATURE_DIR / 'z_test.npy')

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
    test_loader = model.create_dataloader(X_test_scaled, y_test, z_test, batch_size=64, shuffle=False)

    # regression eval
    reg_criterion = torch.nn.HuberLoss(delta=1.5)
    _, mae, rmse = model.test_reg(test_loader, regressor, reg_criterion)

    # compute R^2
    regressor.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for xb, yb, _ in test_loader:
            xb = xb.to(model.device)
            yb = yb.to(model.device)
            preds = regressor(xb).squeeze()
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    r2 = r2_score(all_targets, all_preds)

    # classification eval
    cls_criterion = torch.nn.CrossEntropyLoss()
    _, accuracy, f1, confusion = model.test_cls(test_loader, classifier, cls_criterion)

    # prints
    print("\n" + "=" * 60)
    print("QUICK EVALUATION RESULTS")
    print("=" * 60)
    print(f"Regression → MAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.3f}")
    print(f"Classification → Accuracy: {accuracy:.3f} | Macro-F1: {f1:.3f}")
    print("=" * 60)

    # confusion matrix
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

if __name__ == '__main__':
    main()