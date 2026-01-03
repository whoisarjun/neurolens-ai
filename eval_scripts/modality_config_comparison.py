# Modality configuration comparison script

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ml import model

FEATURE_DIR = Path('models/features')
EVAL_DIR = Path('eval_results')
EVAL_DIR.mkdir(parents=True, exist_ok=True)

N_ACOUSTICS = 52
N_LINGUISTICS = 29
N_SEMANTICS = 18
N_EMBEDDINGS = 1024

lam = float(input('Lambda: '))

def extract_modality_features(X, config):
    # feature indices
    a_start, a_end = 0, N_ACOUSTICS
    l_start, l_end = N_ACOUSTICS, N_ACOUSTICS + N_LINGUISTICS
    s_start, s_end = N_ACOUSTICS + N_LINGUISTICS, N_ACOUSTICS + N_LINGUISTICS + N_SEMANTICS
    e_start, e_end = N_ACOUSTICS + N_LINGUISTICS + N_SEMANTICS, N_ACOUSTICS + N_LINGUISTICS + N_SEMANTICS + N_EMBEDDINGS

    features = []
    if config['A']:
        features.append(X[:, a_start:a_end])
    if config['L']:
        features.append(X[:, l_start:l_end])
    if config['S']:
        features.append(X[:, s_start:s_end])
    if config['E']:
        features.append(X[:, e_start:e_end])

    return np.concatenate(features, axis=1)


def get_feature_counts(config):
    n_a = N_ACOUSTICS if config['A'] else 0
    n_l = N_LINGUISTICS if config['L'] else 0
    n_s = N_SEMANTICS if config['S'] else 0
    n_e = N_EMBEDDINGS if config['E'] else 0
    return n_a, n_l, n_s, n_e


def train_single_config(X_train, X_val, X_test, y_train, y_val, y_test, z_train, z_val, z_test, config, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # extract features for this config
    X_train_mod = extract_modality_features(X_train, config)
    X_val_mod = extract_modality_features(X_val, config)
    X_test_mod = extract_modality_features(X_test, config)

    n_a, n_l, n_s, n_e = get_feature_counts(config)

    # model + dataloaders
    backbone = model.new_backbone(n_acoustics=n_a, n_linguistics=n_l, n_semantics=n_s, n_embeddings=n_e)
    regressor, classifier, reg_criterion, cls_criterion, optimizer, scheduler = model.new_multitask(backbone)

    train_loader = model.create_dataloader(X_train_mod, y_train, z_train, batch_size=64)
    val_loader = model.create_dataloader(X_val_mod, y_val, z_val, batch_size=64, shuffle=False)
    test_loader = model.create_dataloader(X_test_mod, y_test, z_test, batch_size=64, shuffle=False)

    # train
    best_score = -float('inf')
    for epoch in range(50):
        _ = model.train_mt_one_epoch(
            train_loader,
            regressor, classifier,
            reg_criterion, cls_criterion,
            optimizer,
            lam=lam
        )

        val_reg_loss, val_mae, val_rmse = model.test_reg(val_loader, regressor, reg_criterion)
        val_cls_loss, val_acc, val_f1, _ = model.test_cls(val_loader, classifier, cls_criterion)

        scheduler.step(val_reg_loss)

        alpha = 2.0
        score = (-val_mae) + alpha * val_f1

        if score > best_score:
            best_score = score
            best_regressor_state = regressor.state_dict()
            best_classifier_state = classifier.state_dict()

    # best weights
    regressor.load_state_dict(best_regressor_state)
    classifier.load_state_dict(best_classifier_state)

    # test eval
    _, test_mae, test_rmse = model.test_reg(test_loader, regressor, reg_criterion)
    _, test_acc, test_f1, _ = model.test_cls(test_loader, classifier, cls_criterion)

    return test_mae, test_rmse, test_acc, test_f1

def main():
    print("Loading data...")

    # load scaled features and labels
    X_train = np.load(FEATURE_DIR / 'X_train_scaled.npy')
    X_val = np.load(FEATURE_DIR / 'X_val_scaled.npy')
    X_test = np.load(FEATURE_DIR / 'X_test_scaled.npy')
    y_train = np.load(FEATURE_DIR / 'y_train.npy')
    y_val = np.load(FEATURE_DIR / 'y_val.npy')
    y_test = np.load(FEATURE_DIR / 'y_test.npy')
    z_train = np.load(FEATURE_DIR / 'z_train.npy')
    z_val = np.load(FEATURE_DIR / 'z_val.npy')
    z_test = np.load(FEATURE_DIR / 'z_test.npy')

    # configs
    configs = {
        'A only': {'A': True, 'L': False, 'S': False, 'E': False},
        'L only': {'A': False, 'L': True, 'S': False, 'E': False},
        'S only': {'A': False, 'L': False, 'S': True, 'E': False},
        'A+L only': {'A': True, 'L': True, 'S': False, 'E': False},
        'L+S only': {'A': False, 'L': True, 'S': True, 'E': False},
        'A+S only': {'A': True, 'L': False, 'S': True, 'E': False},
        'A+L+S': {'A': True, 'L': True, 'S': True, 'E': False},
        'A+L+S+E': {'A': True, 'L': True, 'S': True, 'E': True}
    }

    results = []

    for config_name, config in configs.items():
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {config_name}")
        print(f"{'=' * 60}")

        maes, rmses, accs, f1s = [], [], [], []

        for seed in tqdm(range(10), desc=f"Training {config_name}"):
            mae, rmse, acc, f1 = train_single_config(
                X_train, X_val, X_test,
                y_train, y_val, y_test,
                z_train, z_val, z_test,
                config, seed
            )
            maes.append(mae)
            rmses.append(rmse)
            accs.append(acc)
            f1s.append(f1)

        avg_mae = np.mean(maes)
        avg_rmse = np.mean(rmses)
        avg_acc = np.mean(accs)
        avg_f1 = np.mean(f1s)

        results.append({
            'Configuration': config_name,
            'MAE': avg_mae,
            'RMSE': avg_rmse,
            'Accuracy': avg_acc,
            'Macro-F1': avg_f1
        })

        print(f"Results: MAE={avg_mae:.3f}, RMSE={avg_rmse:.3f}, Acc={avg_acc:.3f}, F1={avg_f1:.3f}")

    # save
    df = pd.DataFrame(results)
    df.to_csv(EVAL_DIR / 'modality_config_comparison.csv', index=False)

    print("\n" + "=" * 60)
    print("MODALITY CONFIGURATION COMPARISON RESULTS")
    print("=" * 60)
    print(df.to_string(index=False))
    print(f"\nResults saved to {EVAL_DIR / 'modality_config_comparison.csv'}")

if __name__ == '__main__':
    main()