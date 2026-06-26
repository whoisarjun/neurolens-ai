# Paired significance test: full model (A+L+S+E) vs the next-best config (A+L+E)
#
# Backs up the "outperforms all configurations" claim. The modality comparison
# only reports mean ± std per config, which can't tell whether the full model's
# margin over the runner-up is real or noise. Here we retrain both configs over
# the same N_ROUNDS seeds and pair them run-for-run: the test split is fixed for
# every run, and matching seeds give matched initialization/sampling, so run i of
# the full model and run i of A+L+E are a paired observation. We then run a paired
# Wilcoxon signed-rank test (the n=10 recommendation) and a paired t-test on the
# per-seed differences, for MAE (regression) and Accuracy (classification).
#
# Meant to be run from the repo root (e.g. pasted into test.py: `python test.py`).

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

from eval_scripts.eval_stats import format_mean_std, mean_std
from ml import model

FEATURE_DIR = Path('models/features')
EVAL_DIR = Path('eval_results')
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# the two configs being compared. NEXT_BEST is the empirical runner-up from
# modality_config_comparison.csv (A+L+E); change it if your table differs.
FULL_NAME = 'A+L+S+E'
FULL_CONFIG = {'A': True, 'L': True, 'S': True, 'E': True}
NEXT_BEST_NAME = 'A+L+E'
NEXT_BEST_CONFIG = {'A': True, 'L': True, 'S': False, 'E': True}

# significance threshold for the auto-generated wording
ALPHA = 0.05


def require_files(paths):
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        print('\nMissing required files. Train a current model with `python main.py` first:')
        for entry in missing:
            print(f'  - {entry}')
        raise SystemExit(1)


REQUIRED_ARRAYS = [
    f'{name}.npy'
    for split in ('train', 'val', 'test')
    for name in (
        f'X_{split}_scaled', f'y_{split}', f'z_{split}',
        f'y_{split}_mask', f'z_{split}_mask',
    )
]

N_ACOUSTICS = 52
N_LINGUISTICS = 29
N_SEMANTICS = 18
N_EMBEDDINGS = 128  # HuBERT 1024-dim is reduced to 128 via PCA before being saved

# number of matched training runs (seeds) per config
N_ROUNDS = 10


def extract_modality_features(X, config):
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


def train_single_config(
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    z_train, z_val, z_test,
    y_train_mask, y_val_mask, y_test_mask,
    z_train_mask, z_val_mask, z_test_mask,
    config, seed, lam
):
    # identical training procedure to modality_config_comparison.train_single_config,
    # so per-seed numbers are comparable to that table.
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train_mod = extract_modality_features(X_train, config)
    X_val_mod = extract_modality_features(X_val, config)
    X_test_mod = extract_modality_features(X_test, config)

    n_a, n_l, n_s, n_e = get_feature_counts(config)

    backbone = model.new_backbone(n_acoustics=n_a, n_linguistics=n_l, n_semantics=n_s, n_embeddings=n_e)
    regressor, classifier, reg_criterion, cls_criterion, optimizer, scheduler = model.new_multitask(backbone)

    train_loader = model.create_dataloader(X_train_mod, y_train, z_train, y_train_mask, z_train_mask, batch_size=64)
    val_loader = model.create_dataloader(X_val_mod, y_val, z_val, y_val_mask, z_val_mask, batch_size=64, shuffle=False)
    test_loader = model.create_dataloader(X_test_mod, y_test, z_test, y_test_mask, z_test_mask, batch_size=64, shuffle=False)

    best_score = -float('inf')
    for epoch in range(50):
        train_stats = model.train_mt_one_epoch(
            train_loader,
            regressor, classifier,
            reg_criterion, cls_criterion,
            optimizer,
            lam=lam
        )

        val_reg_loss, val_mae, val_rmse, val_reg_score = model.test_reg(
            val_loader, regressor, reg_criterion
        )
        val_cls_loss, val_acc, val_f1, _ = model.test_cls(val_loader, classifier, cls_criterion)

        scheduler_target = val_reg_loss if val_reg_loss is not None else train_stats[1]
        if scheduler_target is not None:
            scheduler.step(scheduler_target)

        score_parts = []
        if val_mae is not None:
            score_parts.append(0.5 * val_reg_score)
        if val_f1 is not None:
            score_parts.append(0.5 * val_f1)
        if not score_parts:
            raise RuntimeError('Validation split has no MMSE or diagnosis labels to score.')
        score = sum(score_parts)

        if score > best_score:
            best_score = score
            best_regressor_state = regressor.state_dict()
            best_classifier_state = classifier.state_dict()

    regressor.load_state_dict(best_regressor_state)
    classifier.load_state_dict(best_classifier_state)

    _, test_mae, test_rmse, _ = model.test_reg(test_loader, regressor, reg_criterion)
    _, test_acc, test_f1, _ = model.test_cls(test_loader, classifier, cls_criterion)

    return test_mae, test_rmse, test_acc, test_f1


def fmt_p(p):
    if p is None or np.isnan(p):
        return 'n/a'
    return '< 0.001' if p < 0.001 else f'{p:.3f}'


def paired_tests(full_vals, next_vals, lower_is_better):
    """Paired Wilcoxon + t-test on per-seed (full - next-best) differences.

    `lower_is_better` is True for MAE, False for accuracy. The one-sided tests use
    the alternative under which the full model *wins*, so a small one-sided p means
    'full is significantly better'. The two-sided p's are the defensible default to
    report; the one-sided p's are included for context.
    """
    full = np.asarray(full_vals, dtype=float)
    nxt = np.asarray(next_vals, dtype=float)
    diff = full - nxt

    # direction in which the full model is the winner
    win_alt = 'less' if lower_is_better else 'greater'
    full_wins = int(np.sum(diff < 0)) if lower_is_better else int(np.sum(diff > 0))
    ties = int(np.sum(diff == 0))

    def safe_wilcoxon(alternative):
        # wilcoxon raises when every difference is zero (no signal to rank)
        try:
            res = stats.wilcoxon(full, nxt, alternative=alternative)
            return float(res.statistic), float(res.pvalue)
        except ValueError:
            return float('nan'), float('nan')

    w_stat, w_p_two = safe_wilcoxon('two-sided')
    _, w_p_one = safe_wilcoxon(win_alt)

    t_two = stats.ttest_rel(full, nxt)
    t_one = stats.ttest_rel(full, nxt, alternative=win_alt)

    std_diff = float(np.std(diff, ddof=1))
    cohens_dz = float(np.mean(diff) / std_diff) if std_diff > 0 else float('nan')

    return {
        'full_mean': float(np.mean(full)), 'full_std': float(np.std(full)),
        'next_mean': float(np.mean(nxt)), 'next_std': float(np.std(nxt)),
        'mean_diff': float(np.mean(diff)), 'median_diff': float(np.median(diff)),
        'full_wins': full_wins, 'ties': ties, 'n': int(full.size),
        'cohens_dz': cohens_dz,
        'wilcoxon_stat': w_stat,
        'wilcoxon_p_two_sided': w_p_two, 'wilcoxon_p_one_sided': w_p_one,
        'ttest_t': float(t_two.statistic),
        'ttest_p_two_sided': float(t_two.pvalue), 'ttest_p_one_sided': float(t_one.pvalue),
    }


def main():
    print("Loading data...")

    require_files([FEATURE_DIR / name for name in REQUIRED_ARRAYS])

    lam = float(input('Lambda: '))

    X_train = np.load(FEATURE_DIR / 'X_train_scaled.npy')
    X_val = np.load(FEATURE_DIR / 'X_val_scaled.npy')
    X_test = np.load(FEATURE_DIR / 'X_test_scaled.npy')
    y_train = np.load(FEATURE_DIR / 'y_train.npy')
    y_val = np.load(FEATURE_DIR / 'y_val.npy')
    y_test = np.load(FEATURE_DIR / 'y_test.npy')
    z_train = np.load(FEATURE_DIR / 'z_train.npy')
    z_val = np.load(FEATURE_DIR / 'z_val.npy')
    z_test = np.load(FEATURE_DIR / 'z_test.npy')
    y_train_mask = np.load(FEATURE_DIR / 'y_train_mask.npy')
    y_val_mask = np.load(FEATURE_DIR / 'y_val_mask.npy')
    y_test_mask = np.load(FEATURE_DIR / 'y_test_mask.npy')
    z_train_mask = np.load(FEATURE_DIR / 'z_train_mask.npy')
    z_val_mask = np.load(FEATURE_DIR / 'z_val_mask.npy')
    z_test_mask = np.load(FEATURE_DIR / 'z_test_mask.npy')

    # match training: inverse-frequency MMSE weights from train labels, as in main.py
    model.set_mmse_freq(np.asarray(y_train)[np.asarray(y_train_mask)])

    configs = {FULL_NAME: FULL_CONFIG, NEXT_BEST_NAME: NEXT_BEST_CONFIG}

    # per-seed metrics, kept paired by seed across the two configs
    per_seed = {name: {'mae': [], 'acc': []} for name in configs}

    for seed in tqdm(range(N_ROUNDS), desc='Matched runs'):
        for name, config in configs.items():
            mae, rmse, acc, f1 = train_single_config(
                X_train, X_val, X_test,
                y_train, y_val, y_test,
                z_train, z_val, z_test,
                y_train_mask, y_val_mask, y_test_mask,
                z_train_mask, z_val_mask, z_test_mask,
                config, seed, lam
            )
            per_seed[name]['mae'].append(mae)
            per_seed[name]['acc'].append(acc)

    # raw matched runs (one row per seed) — the "10 matched runs" the test is built on
    runs_df = pd.DataFrame({
        'seed': list(range(N_ROUNDS)),
        f'{FULL_NAME}_MAE': per_seed[FULL_NAME]['mae'],
        f'{NEXT_BEST_NAME}_MAE': per_seed[NEXT_BEST_NAME]['mae'],
        'MAE_diff(full-next)': np.subtract(per_seed[FULL_NAME]['mae'], per_seed[NEXT_BEST_NAME]['mae']),
        f'{FULL_NAME}_Accuracy': per_seed[FULL_NAME]['acc'],
        f'{NEXT_BEST_NAME}_Accuracy': per_seed[NEXT_BEST_NAME]['acc'],
        'Accuracy_diff(full-next)': np.subtract(per_seed[FULL_NAME]['acc'], per_seed[NEXT_BEST_NAME]['acc']),
    })
    runs_df.to_csv(EVAL_DIR / 'paired_significance_runs.csv', index=False)

    # paired tests per metric (MAE: lower is better, Accuracy: higher is better)
    mae_res = paired_tests(per_seed[FULL_NAME]['mae'], per_seed[NEXT_BEST_NAME]['mae'], lower_is_better=True)
    acc_res = paired_tests(per_seed[FULL_NAME]['acc'], per_seed[NEXT_BEST_NAME]['acc'], lower_is_better=False)

    summary_df = pd.DataFrame([
        {'Metric': 'MAE', 'Better': 'lower', **mae_res},
        {'Metric': 'Accuracy', 'Better': 'higher', **acc_res},
    ])
    summary_df.to_csv(EVAL_DIR / 'paired_significance_test.csv', index=False)

    # ---- console report ----
    print("\n" + "=" * 72)
    print(f"PAIRED SIGNIFICANCE TEST  ({FULL_NAME} vs {NEXT_BEST_NAME}, n={N_ROUNDS} matched runs)")
    print("=" * 72)

    for metric, res in (('MAE', mae_res), ('Accuracy', acc_res)):
        print(f"\n{metric} (lower is better)" if metric == 'MAE' else f"\n{metric} (higher is better)")
        print(f"  {FULL_NAME:>8}: {format_mean_std(res['full_mean'], res['full_std'])}")
        print(f"  {NEXT_BEST_NAME:>8}: {format_mean_std(res['next_mean'], res['next_std'])}")
        print(f"  mean diff (full - next): {res['mean_diff']:+.4f}   "
              f"full wins {res['full_wins']}/{res['n']} runs (ties: {res['ties']})")
        print(f"  Wilcoxon signed-rank:  p(two-sided) = {fmt_p(res['wilcoxon_p_two_sided'])}   "
              f"p(one-sided, full better) = {fmt_p(res['wilcoxon_p_one_sided'])}")
        print(f"  Paired t-test:         p(two-sided) = {fmt_p(res['ttest_p_two_sided'])}   "
              f"p(one-sided, full better) = {fmt_p(res['ttest_p_one_sided'])}")
        print(f"  Effect size (Cohen's dz): {res['cohens_dz']:+.3f}")

    # ---- auto-generated wording (Wilcoxon two-sided, the recommended primary test) ----
    mae_p = mae_res['wilcoxon_p_two_sided']
    acc_p = acc_res['wilcoxon_p_two_sided']
    mae_sig = not np.isnan(mae_p) and mae_p < ALPHA
    acc_sig = not np.isnan(acc_p) and acc_p < ALPHA

    print("\n" + "-" * 72)
    print("SUGGESTED WORDING (based on two-sided Wilcoxon, alpha = 0.05)")
    print("-" * 72)
    if mae_sig and acc_sig:
        print(
            f"...achieved the strongest mean performance on both tasks. A paired Wilcoxon "
            f"signed-rank test across the {N_ROUNDS} matched runs confirmed that the full model "
            f"significantly outperformed the next-best configuration ({NEXT_BEST_NAME.replace('+', ' + ')}) "
            f"on MAE (p = {fmt_p(mae_p)}) and accuracy (p = {fmt_p(acc_p)})."
        )
    elif not mae_sig and not acc_sig:
        print(
            f"...achieved the best mean performance on both tasks. A paired Wilcoxon signed-rank "
            f"test across the {N_ROUNDS} matched runs indicated that the margin over the next-best "
            f"configuration ({NEXT_BEST_NAME.replace('+', ' + ')}) did not reach significance "
            f"(MAE p = {fmt_p(mae_p)}; accuracy p = {fmt_p(acc_p)}); the full model thus delivers the "
            f"best point estimate while performing comparably to the strongest partial configuration."
        )
    else:
        sig_metric = 'MAE' if mae_sig else 'accuracy'
        ns_metric = 'accuracy' if mae_sig else 'MAE'
        print(
            f"...achieved the best mean performance on both tasks. A paired Wilcoxon signed-rank test "
            f"across the {N_ROUNDS} matched runs found the full model's advantage over the next-best "
            f"configuration ({NEXT_BEST_NAME.replace('+', ' + ')}) significant on {sig_metric} "
            f"(p = {fmt_p(mae_p if mae_sig else acc_p)}) but not on {ns_metric} "
            f"(p = {fmt_p(acc_p if mae_sig else mae_p)})."
        )
    print("(Replace the p-values / claim in the paper with the line above; the leave-one-out "
          "ablation still carries the 'all modalities matter' claim independently.)")

    print(f"\nSaved per-run table to {EVAL_DIR / 'paired_significance_runs.csv'}")
    print(f"Saved test summary to {EVAL_DIR / 'paired_significance_test.csv'}")


if __name__ == '__main__':
    main()
