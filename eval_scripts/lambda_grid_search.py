# Lambda grid search script

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


def train_with_lambda(X_train, X_val, X_test, y_train, y_val, y_test, z_train, z_val, z_test, lam, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # models
    backbone = model.new_backbone()
    regressor, classifier, reg_criterion, cls_criterion, optimizer, scheduler = model.new_multitask(backbone)

    # dataloaders
    train_loader = model.create_dataloader(X_train, y_train, z_train, batch_size=64)
    val_loader = model.create_dataloader(X_val, y_val, z_val, batch_size=64, shuffle=False)
    test_loader = model.create_dataloader(X_test, y_test, z_test, batch_size=64, shuffle=False)

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

    # load best weights
    regressor.load_state_dict(best_regressor_state)
    classifier.load_state_dict(best_classifier_state)

    # eval on test
    _, test_mae, test_rmse = model.test_reg(test_loader, regressor, reg_criterion)
    _, test_acc, test_f1, _ = model.test_cls(test_loader, classifier, cls_criterion)

    return test_mae, test_rmse, test_acc, test_f1


def evaluate_lambda(X_train, X_val, X_test, y_train, y_val, y_test, z_train, z_val, z_test, lam):
    maes, rmses, accs, f1s = [], [], [], []

    for seed in tqdm(range(10), desc=f"Lambda {lam:.2f}", leave=False):
        mae, rmse, acc, f1 = train_with_lambda(
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            z_train, z_val, z_test,
            lam, seed
        )
        maes.append(mae)
        rmses.append(rmse)
        accs.append(acc)
        f1s.append(f1)

    avg_mae = np.mean(maes)
    avg_rmse = np.mean(rmses)
    avg_acc = np.mean(accs)
    avg_f1 = np.mean(f1s)

    # calc score
    alpha = 2.0
    score = (-avg_mae) + alpha * avg_f1

    return avg_mae, avg_rmse, avg_acc, avg_f1, score


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

    all_results = []

    # coarse search
    print("\n" + "=" * 60)
    print("LEVEL 1: Coarse Grid Search")
    print("=" * 60)

    level1_lambdas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    level1_results = []

    for lam in level1_lambdas:
        print(f"\nEvaluating lambda = {lam:.1f}")
        mae, rmse, acc, f1, score = evaluate_lambda(
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            z_train, z_val, z_test,
            lam
        )

        result = {
            'Lambda': lam,
            'MAE': mae,
            'RMSE': rmse,
            'Accuracy': acc,
            'Macro-F1': f1,
            'Score': score
        }
        level1_results.append(result)
        all_results.append(result)

        print(f"Results: MAE={mae:.3f}, RMSE={rmse:.3f}, Acc={acc:.3f}, F1={f1:.3f}, Score={score:.3f}")

    # best lambda cat
    best_level1 = max(level1_results, key=lambda x: x['Score'])
    best_lambda_l1 = best_level1['Lambda']
    print(f"\nBest Level 1 Lambda: {best_lambda_l1:.1f} (Score: {best_level1['Score']:.3f})")

    # fine search around best lambda
    print("\n" + "=" * 60)
    print("LEVEL 2: Fine Grid Search")
    print("=" * 60)

    # generate fine grid around best lambda
    level2_lambdas = [best_lambda_l1 + i * 0.01 for i in range(10)]  # e.g., 0.30 to 0.39

    level2_results = []

    for lam in level2_lambdas:
        print(f"\nEvaluating lambda = {lam:.2f}")
        mae, rmse, acc, f1, score = evaluate_lambda(
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            z_train, z_val, z_test,
            lam
        )

        result = {
            'Lambda': lam,
            'MAE': mae,
            'RMSE': rmse,
            'Accuracy': acc,
            'Macro-F1': f1,
            'Score': score
        }
        level2_results.append(result)
        all_results.append(result)

        print(f"Results: MAE={mae:.3f}, RMSE={rmse:.3f}, Acc={acc:.3f}, F1={f1:.3f}, Score={score:.3f}")

    # find overall best
    all_results_sorted = sorted(all_results, key=lambda x: x['Score'], reverse=True)
    top_5 = all_results_sorted[:5]
    best_lambda = top_5[0]['Lambda']

    # save all results
    df = pd.DataFrame(all_results)
    df = df.sort_values('Score', ascending=False)
    df.to_csv(EVAL_DIR / 'lambda_grid_search.csv', index=False)

    print("\n" + "=" * 60)
    print("TOP 5 LAMBDA CONFIGURATIONS")
    print("=" * 60)

    top5_df = pd.DataFrame(top_5)
    top5_df.insert(0, 'Rank', range(1, len(top5_df) + 1))
    top5_df.loc[0, 'Rank'] = '🏆 1'

    top5_df = top5_df[
        ['Rank', 'Lambda', 'MAE', 'RMSE', 'Accuracy', 'Macro-F1', 'Score']
    ].round(3)

    print(top5_df.to_string(index=False))

    print(f"\n{'=' * 60}")
    print(f"All results saved to {EVAL_DIR / 'lambda_grid_search.csv'}")


if __name__ == '__main__':
    main()