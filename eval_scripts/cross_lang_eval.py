# Cross-language evaluation script

import json
import sys
from pathlib import Path

# allow `python eval_scripts/<name>.py` from the repo root to import project packages
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from eval_scripts.eval_stats import bootstrap_classification, bootstrap_regression, format_mean_std
from ml import model
from utils.language import normalize_language

# bootstrap resamples used to estimate the mean ± std of each metric
N_BOOTSTRAP = 1000


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

TEST_JSON = Path('data_jsons/test.json')
REG_WEIGHTS_PATH = Path('models/model_weights_reg.pth')
CLS_WEIGHTS_PATH = Path('models/model_weights_cls.pth')
SCALER_PATH = Path('models/model_scaler.pkl')
FEATURE_DIR = Path('models/features')
EVAL_DIR = Path('eval_results')
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_subset(name, mask, reg_preds, cls_preds, y, z, y_mask, z_mask):
    # restrict to subset samples that also carry the relevant label
    reg_sel = mask & y_mask
    cls_sel = mask & z_mask

    reg_stats = bootstrap_regression(reg_preds[reg_sel], y[reg_sel], N_BOOTSTRAP)
    cls_stats = bootstrap_classification(z[cls_sel], cls_preds[cls_sel], N_BOOTSTRAP)

    row = {
        'Split': name,
        'Samples': int(mask.sum()),
        'MMSE labeled': int(reg_sel.sum()),
        'Diagnosis labeled': int(cls_sel.sum()),
    }
    for metric in ('MAE', 'RMSE', 'R²'):
        row[f'{metric}_mean'], row[f'{metric}_std'] = reg_stats[metric]
    for metric in ('Accuracy', 'Macro-F1'):
        row[f'{metric}_mean'], row[f'{metric}_std'] = cls_stats[metric]
    return row


def main():
    print('Loading test split metadata...')

    require_files([
        TEST_JSON,
        FEATURE_DIR / 'X_test_scaled.npy', FEATURE_DIR / 'y_test.npy',
        FEATURE_DIR / 'z_test.npy', FEATURE_DIR / 'y_test_mask.npy',
        FEATURE_DIR / 'z_test_mask.npy',
        REG_WEIGHTS_PATH, CLS_WEIGHTS_PATH, SCALER_PATH,
    ])

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
    y_test_mask = np.load(FEATURE_DIR / 'y_test_mask.npy').astype(bool)
    z_test_mask = np.load(FEATURE_DIR / 'z_test_mask.npy').astype(bool)

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

    load_weights((REG_WEIGHTS_PATH, regressor), (CLS_WEIGHTS_PATH, classifier))

    # inference is deterministic; run one forward pass and bootstrap per subset
    regressor.eval()
    classifier.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(model.device)
        reg_preds = regressor(X_tensor).squeeze(-1).cpu().numpy()
        cls_preds = torch.argmax(classifier(X_tensor), dim=1).cpu().numpy()

    subset_masks = {
        'English only': languages == 'en',
        'Mandarin only': languages == 'zh',
        'All test data': np.ones(len(languages), dtype=bool),
    }

    results = [
        evaluate_subset(
            subset_name, subset_mask,
            reg_preds, cls_preds,
            y_test, z_test, y_test_mask, z_test_mask,
        )
        for subset_name, subset_mask in subset_masks.items()
    ]

    df = pd.DataFrame(results)
    csv_path = EVAL_DIR / 'cross_lang_eval.csv'
    df.to_csv(csv_path, index=False)

    # console view: collapse each metric's mean/std columns into "mean ± std"
    display = df[['Split', 'Samples', 'MMSE labeled', 'Diagnosis labeled']].copy()
    for metric in ('MAE', 'RMSE', 'R²', 'Accuracy', 'Macro-F1'):
        display[metric] = [
            format_mean_std(m, s) for m, s in zip(df[f'{metric}_mean'], df[f'{metric}_std'])
        ]

    print('\n' + '=' * 72)
    print(f'CROSS-LANGUAGE TEST EVALUATION (bootstrap mean ± std, {N_BOOTSTRAP} resamples)')
    print('=' * 72)
    print(display.to_string(index=False))
    print(f'\nResults saved to {csv_path}')


if __name__ == '__main__':
    main()
