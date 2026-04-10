# Cross-language evaluation script

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

from ml import model
from utils.language import normalize_language

TEST_JSON = Path('data_jsons/test.json')
REG_WEIGHTS_PATH = Path('models/model_weights_reg.pth')
CLS_WEIGHTS_PATH = Path('models/model_weights_cls.pth')
SCALER_PATH = Path('models/model_scaler.pkl')
FEATURE_DIR = Path('models/features')
EVAL_DIR = Path('eval_results')
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def compute_r2(loader, regressor):
    regressor.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for xb, yb, _, y_mask, _ in loader:
            if not y_mask.any():
                continue

            xb = xb[y_mask].to(model.device)
            yb = yb[y_mask].to(model.device)
            batch_preds = regressor(xb).squeeze(-1)

            preds.append(batch_preds.cpu().numpy())
            targets.append(yb.squeeze(-1).cpu().numpy())

    if not preds:
        return None

    return r2_score(np.concatenate(targets), np.concatenate(preds))


def subset_arrays(mask, X, y, z, y_mask, z_mask):
    return (
        X[mask],
        y[mask],
        z[mask],
        y_mask[mask],
        z_mask[mask],
    )


def evaluate_subset(name, mask, X, y, z, y_mask, z_mask, regressor, classifier):
    X_sub, y_sub, z_sub, y_mask_sub, z_mask_sub = subset_arrays(mask, X, y, z, y_mask, z_mask)

    loader = model.create_dataloader(
        X_sub,
        y_sub,
        z_sub,
        y_mask_sub,
        z_mask_sub,
        batch_size=64,
        shuffle=False,
    )

    reg_criterion = torch.nn.HuberLoss(delta=1.5)
    cls_criterion = torch.nn.CrossEntropyLoss()

    _, mae, rmse = model.test_reg(loader, regressor, reg_criterion)
    _, accuracy, f1, _ = model.test_cls(loader, classifier, cls_criterion)
    r2 = compute_r2(loader, regressor) if mae is not None else None

    return {
        'Split': name,
        'Samples': int(mask.sum()),
        'MMSE labeled': int(np.sum(y_mask_sub)),
        'Diagnosis labeled': int(np.sum(z_mask_sub)),
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'Accuracy': accuracy,
        'Macro-F1': f1,
    }


def main():
    print('Loading test split metadata...')

    with TEST_JSON.open('r', encoding='utf-8') as f:
        test_data = json.load(f)['data']

    languages = np.array([
        normalize_language(item.get('language'))
        for item in test_data
    ])

    print('Loading saved test features...')
    X_test_scaled = np.load(FEATURE_DIR / 'X_test_scaled.npy')
    y_test = np.load(FEATURE_DIR / 'y_test.npy')
    z_test = np.load(FEATURE_DIR / 'z_test.npy')
    y_test_mask = np.load(FEATURE_DIR / 'y_test_mask.npy')
    z_test_mask = np.load(FEATURE_DIR / 'z_test_mask.npy')

    if len(languages) != len(X_test_scaled):
        raise ValueError(
            'Mismatch between test.json entries and saved test features: '
            f'{len(languages)} vs {len(X_test_scaled)}'
        )

    print('Loading trained models...')
    model.load_scaler(SCALER_PATH)

    backbone = model.new_backbone()
    regressor = model.MMSERegression(backbone).to(model.device)
    classifier = model.CognitiveStatusClassification(backbone).to(model.device)

    model.load(REG_WEIGHTS_PATH, regressor)
    model.load(CLS_WEIGHTS_PATH, classifier)

    subset_masks = {
        'English only': languages == 'en',
        'Mandarin only': languages == 'zh',
        'All test data': np.ones(len(languages), dtype=bool),
    }

    results = []
    for subset_name, subset_mask in subset_masks.items():
        results.append(
            evaluate_subset(
                subset_name,
                subset_mask,
                X_test_scaled,
                y_test,
                z_test,
                y_test_mask,
                z_test_mask,
                regressor,
                classifier,
            )
        )

    df = pd.DataFrame(results)
    csv_path = EVAL_DIR / 'cross_lang_eval.csv'
    df.to_csv(csv_path, index=False)

    print('\n' + '=' * 72)
    print('CROSS-LANGUAGE TEST EVALUATION')
    print('=' * 72)
    print(df.to_string(index=False, float_format=lambda x: f'{x:.3f}' if pd.notna(x) else 'n/a'))
    print(f'\nResults saved to {csv_path}')


if __name__ == '__main__':
    main()
