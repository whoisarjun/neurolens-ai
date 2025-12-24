# Script to investigate whether multimodal analysis outperforms unimodal models

from pathlib import Path

import numpy as np

from main import process_split, train
from ml import model

use_cache_list = {
    'transcript': True,
    'acoustics': True,
    'linguistics': True,
    'semantics': True,
    'embeddings': True
}

RESET = '\033[0m'
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
CYAN = '\033[36m'
MAGENTA = '\033[35m'
BOLD = '\033[1m'

TRAIN_JSON = Path('data_jsons/train.json')
VAL_JSON = Path('data_jsons/val.json')
TEST_JSON = Path('data_jsons/test.json')

X_train, y_train, z_train = process_split(TRAIN_JSON, 'train', use_cache=use_cache_list, augment=True)
X_val, y_val, z_val = process_split(VAL_JSON, 'validation', use_cache=use_cache_list, augment=False)
X_test, y_test, z_test = process_split(TEST_JSON, 'test', use_cache=use_cache_list, augment=False)
print(f"\n{GREEN}Done processing all data!{RESET}")

# scale features (fit on train only)
print(f"\n{BOLD}{CYAN}Fitting scaler on train data...{RESET}")
model.fit_scaler(X_train)

X_train_scaled = model.transform_features(X_train)
X_val_scaled = model.transform_features(X_val)
X_test_scaled = model.transform_features(X_test)

options = [
    ('Acoustics only: ', (52, 0, 0, 0)),
    ('Linguistics only: ', (0, 29, 0, 0)),
    ('Semantics only: ', (0, 0, 18, 0)),
    ('A+L only: ', (52, 29, 0, 0)),
    ('L+S only: ', (0, 29, 18, 0)),
    ('A+S only: ', (52, 0, 18, 0)),
    ('Without embeddings: ', (52, 29, 18, 0)),
    ('Normal (everything): ', (52, 29, 18, 1024))
]

for name, nums in options:
    print(f'Run: {name}...')

    a, l, s, e = nums

    # feature slicing based on enabled modalities
    idx = 0
    slices = []

    if a > 0:
        slices.append(slice(idx, idx + 52))
        idx += 52
    else:
        idx += 52

    if l > 0:
        slices.append(slice(idx, idx + 29))
        idx += 29
    else:
        idx += 29

    if s > 0:
        slices.append(slice(idx, idx + 18))
        idx += 18
    else:
        idx += 18

    if e > 0:
        slices.append(slice(idx, idx + 1024))
        idx += 1024
    else:
        idx += 1024

    def apply_slices(X):
        return np.concatenate([X[:, sl] for sl in slices], axis=1)

    Xtr = apply_slices(X_train_scaled)
    Xva = apply_slices(X_val_scaled)
    Xte = apply_slices(X_test_scaled)

    backbone = model.new_backbone(
        n_acoustics=a, n_linguistics=l, n_semantics=s, n_embeddings=e
    )
    regressor, reg_criterion, reg_optimizer, scheduler = model.new_regressor(backbone)
    classifier, cls_criterion, cls_optimizer = model.new_classifier(backbone)

    test_mae, test_rmse, test_acc, test_f1, test_cm = train(
        Xtr, Xva, Xte,
        y_train, y_val, y_test,
        z_train, z_val, z_test,
        regressor, reg_criterion, reg_optimizer, scheduler,
        classifier, cls_criterion, cls_optimizer,
        verbose=False
    )

    print(f'MAE: {test_mae}')
    print(f'RMSE: {test_rmse}')
    print(f'Acc: {test_acc}')
    print(f'F1: {test_f1}')
    print(f'Conf: {test_cm}\n\n')





