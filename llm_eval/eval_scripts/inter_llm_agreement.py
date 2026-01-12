from pathlib import Path
import json
from itertools import combinations
import matplotlib.pyplot as plt

import numpy as np
from features import semantics

# Ensure output directory exists
eval_dir = Path('eval_results')
eval_dir.mkdir(parents=True, exist_ok=True)

def icc2_1(ratings: np.ndarray) -> float:
    """Compute ICC(2,1) for a ratings matrix of shape (n_targets, n_raters)."""
    ratings = np.asarray(ratings, dtype=float)
    if ratings.ndim != 2:
        raise ValueError("ratings must be a 2D array of shape (n_targets, n_raters)")

    n, k = ratings.shape
    if n <= 1 or k <= 1:
        return float("nan")

    mean_per_target = ratings.mean(axis=1, keepdims=True)
    mean_per_rater = ratings.mean(axis=0, keepdims=True)
    grand_mean = ratings.mean()

    # Between-targets mean square (rows)
    MSR = k * ((mean_per_target - grand_mean) ** 2).sum() / (n - 1)

    # Between-raters mean square (columns)
    if k > 1:
        MSC = n * ((mean_per_rater - grand_mean) ** 2).sum() / (k - 1)
    else:
        MSC = 0.0

    # Residual mean square
    residual = (ratings - mean_per_target - mean_per_rater + grand_mean) ** 2
    if (n - 1) * (k - 1) > 0:
        MSE = residual.sum() / ((n - 1) * (k - 1))
    else:
        MSE = 0.0

    denom = MSR + (k - 1) * MSE + (k * (MSC - MSE) / n)
    if denom == 0:
        return 0.0

    icc = (MSR - MSE) / denom
    return float(icc)

files = ['deepseek', 'granite', 'llama', 'ministral', 'qwen']

print('\n' + '=' * 60)
print('INTER-LLM AGREEMENT')
print('=' * 60)

if hasattr(semantics, 'all_features'):
    FEATURE_NAMES = semantics.all_features
    NUMBER_OF_FEATURES = len(FEATURE_NAMES)
else:
    NUMBER_OF_FEATURES = 18
    FEATURE_NAMES = [f'Feature {i + 1}' for i in range(NUMBER_OF_FEATURES)]

# load data and build per-llm score vectors

model_flat_vectors = {}
model_feature_matrices = {}

for f in files:
    with Path(f'models/features/{f}.json').open() as j:
        raw = json.load(j)
        results = [r.get('results') for r in raw.get('results')]
    transcript_means = []
    for r in results:
        arr = np.array(r, dtype=float)
        mean_over_runs = arr.mean(axis=0)
        transcript_means.append(mean_over_runs)

    mat = np.stack(transcript_means, axis=0)
    model_feature_matrices[f] = mat
    model_flat_vectors[f] = mat.reshape(-1)

models = files
n_models = len(models)

# Overall inter-LLM ICC(2,1) across all transcripts × features
all_scores = np.stack([model_flat_vectors[m] for m in models], axis=1)  # shape: (n_targets, n_models)
overall_icc = icc2_1(all_scores)

print('\nA. Overall inter-LLM ICC(2,1) across all transcripts × features')
print(f'Overall ICC(2,1): {overall_icc:.3f}')

# 5x5 pairwise inter-LLM ICC(2,1) matrix
icc_matrix = np.zeros((n_models, n_models), dtype=float)

for i, m1 in enumerate(models):
    v1 = model_flat_vectors[m1]
    for j, m2 in enumerate(models):
        if i == j:
            icc_matrix[i, j] = 1.0
            continue
        v2 = model_flat_vectors[m2]
        ratings = np.stack([v1, v2], axis=1)  # shape: (n_targets, 2)
        icc_matrix[i, j] = icc2_1(ratings)

print('\nB. Pairwise inter-LLM ICC(2,1) Matrix (across all transcripts × features)')
header = ' ' * 10 + ''.join(f'{m:>12}' for m in models)
print(header)
for i, m1 in enumerate(models):
    row_vals = ''.join(f'{icc_matrix[i, j]:12.3f}' for j in range(n_models))
    print(f'{m1:>10}{row_vals}')

# 5x5 mean absolute difference

mad_matrix = np.zeros((n_models, n_models), dtype=float)

for i, m1 in enumerate(models):
    v1 = model_flat_vectors[m1]
    for j, m2 in enumerate(models):
        v2 = model_flat_vectors[m2]
        mad = float(np.mean(np.abs(v1 - v2)))
        mad_matrix[i, j] = mad

# 5x5 MAD heatmap
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(mad_matrix, vmin=0.0, vmax=np.max(mad_matrix))

ax.set_xticks(range(n_models))
ax.set_yticks(range(n_models))
ax.set_xticklabels(models, rotation=45, ha='right')
ax.set_yticklabels(models)

# Annotate each cell with MAD value
for i in range(n_models):
    for j in range(n_models):
        ax.text(j, i, f'{mad_matrix[i, j]:.3f}',
                ha='center', va='center', fontsize=8)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Mean Absolute Difference', rotation=90)
ax.set_title('Inter-LLM MAD Matrix')
fig.tight_layout()
fig.savefig(eval_dir / 'inter_llm_mad_matrix.png', dpi=300)
plt.close(fig)

print('\nC. Mean Absolute Difference (MAD) Matrix (across all transcripts × features)')
header = ' ' * 10 + ''.join(f'{m:>12}' for m in models)
print(header)
for i, m1 in enumerate(models):
    row_vals = ''.join(f'{mad_matrix[i, j]:12.3f}' for j in range(n_models))
    print(f'{m1:>10}{row_vals}')

# 5x5 inter-LLM Pearson correlation heatmap
# all_scores has shape (n_targets, n_models), so we correlate columns
corr_matrix = np.corrcoef(all_scores, rowvar=False)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(corr_matrix, vmin=-1.0, vmax=1.0)

ax.set_xticks(range(n_models))
ax.set_yticks(range(n_models))
ax.set_xticklabels(models, rotation=45, ha='right')
ax.set_yticklabels(models)

# Annotate each cell with the correlation value
for i in range(n_models):
    for j in range(n_models):
        ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                ha='center', va='center', fontsize=8)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Pearson r', rotation=90)
ax.set_title('Inter-LLM Correlation Heatmap')
fig.tight_layout()
fig.savefig(eval_dir / 'inter_llm_corr_heatmap.png', dpi=300)
plt.close(fig)

# feature-level agreement

print('\nD. Feature-level inter-LLM ICC(2,1) across all transcripts')

for feat_idx in range(NUMBER_OF_FEATURES):
    # Build ratings matrix: rows = transcripts, columns = LLMs
    per_llm_feature = [model_feature_matrices[m][:, feat_idx] for m in models]
    ratings = np.stack(per_llm_feature, axis=1)  # shape: (n_transcripts, n_models)

    icc_feat = icc2_1(ratings)
    fname = FEATURE_NAMES[feat_idx] if feat_idx < len(FEATURE_NAMES) else f'Feature {feat_idx + 1}'
    print(f'{feat_idx:2d}  {fname:40s}: {icc_feat:.3f}')
