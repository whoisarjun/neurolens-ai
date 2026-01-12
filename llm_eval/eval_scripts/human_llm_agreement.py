from pathlib import Path
from glob import glob
import json
from itertools import combinations

import numpy as np
from features import semantics
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt

# config

LLM_MODELS = ['deepseek', 'granite', 'llama', 'ministral', 'qwen']
HUMAN_EVAL_DIR = Path('human_eval')

# Overall intra-LLM ICC(3,k) per model (from intra_llm_consistency.py)
LLM_ICC = {
    'deepseek': 0.9916343,
    'granite': 0.9981859,
    'llama': 0.9963090,
    'ministral': 0.9961227,
    'qwen': 0.9967810,
}

print('\n' + '=' * 60)
print('INTER-LLM + HUMAN AGREEMENT')
print('=' * 60)

if hasattr(semantics, 'all_features'):
    FEATURE_NAMES = list(semantics.all_features)
    NUMBER_OF_FEATURES = len(FEATURE_NAMES)
else:
    NUMBER_OF_FEATURES = 18
    FEATURE_NAMES = [f'Feature {i + 1}' for i in range(NUMBER_OF_FEATURES)]

# utils

def cohen_kappa(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("Shapes of a and b must match for Cohen's kappa")

    labels = np.union1d(a, b)
    n_cats = len(labels)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    conf = np.zeros((n_cats, n_cats), dtype=float)
    for x, y in zip(a, b):
        conf[label_to_idx[x], label_to_idx[y]] += 1.0

    n = conf.sum()
    if n == 0:
        return 0.0

    p_o = np.trace(conf) / n
    p_a = conf.sum(axis=1) / n
    p_b = conf.sum(axis=0) / n
    p_e = float(np.dot(p_a, p_b))

    if 1.0 - p_e == 0.0:
        return 0.0
    return (p_o - p_e) / (1.0 - p_e)

def safe_pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def safe_spearman(x, y):
    """
    Spearman's rho for ordinal 0–4 data; returns 0.0 if one side is constant.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.std() == 0 or y.std() == 0:
        return 0.0
    rx = pd.Series(x).rank(method='average')
    ry = pd.Series(y).rank(method='average')
    return float(np.corrcoef(rx, ry)[0, 1])

# load llm scores

llm_scores = {}

for model in LLM_MODELS:
    path = Path('models/features') / f'{model}.json'
    with path.open() as j:
        raw = json.load(j)
        entries = raw.get('results', [])

    model_dict = {}
    for entry in entries:
        fname = Path(entry['file']).name.lower()
        runs = np.asarray(entry['results'], dtype=float)
        mean_over_runs = runs.mean(axis=0)
        model_dict[fname] = mean_over_runs

    llm_scores[model] = model_dict

llm_file_sets = [set(d.keys()) for d in llm_scores.values()]
common_llm_files = set.intersection(*llm_file_sets)

# human scores

human_json_paths = sorted(HUMAN_EVAL_DIR.glob('*.json'))
human_scores = []

for hp in human_json_paths:
    with hp.open() as j:
        raw = json.load(j)
        entries = raw.get('results', [])
    h_dict = {}
    for entry in entries:
        fname = Path(entry['file']).name.lower()
        scores = np.asarray(entry['results'], dtype=float)
        h_dict[fname] = scores
    human_scores.append(h_dict)

if len(human_scores) != 3:
    print(f'WARNING: Expected 3 human raters, found {len(human_scores)}')

human_file_sets = [set(d.keys()) for d in human_scores]
common_human_files = set.intersection(*human_file_sets)

print("Human JSON paths:", human_json_paths)
print("Num LLM files per model:", {m: len(d) for m, d in llm_scores.items()})
print("Num files per human rater:", [len(d) for d in human_scores])

common_files = sorted(common_llm_files.intersection(common_human_files))
n_files = len(common_files)

print(f'\nUsing {n_files} files common to all LLMs and human raters.\n')

# build aligned matrices
llm_mats = {}
for model in LLM_MODELS:
    mat = np.stack([llm_scores[model][f] for f in common_files], axis=0)
    llm_mats[model] = mat

human_mats = []
for h_dict in human_scores:
    mat = np.stack([h_dict[f] for f in common_files], axis=0)
    human_mats.append(mat)

human_mats = np.stack(human_mats, axis=0)
n_humans = human_mats.shape[0]

human_mean = human_mats.mean(axis=0)

# human inter-rater reliability

print('\n' + '=' * 60)
print('A. HUMAN INTER-RATER RELIABILITY (ICC)')
print('=' * 60)

# Build DataFrame for ICC: each row = one score from (file, rater, feature)
rows = []
for rater_idx, h_mat in enumerate(human_mats):
    for file_idx, fname in enumerate(common_files):
        for feat_idx in range(NUMBER_OF_FEATURES):
            rows.append({
                "targets": file_idx,
                "raters": rater_idx,
                "feature": feat_idx,
                "score": float(h_mat[file_idx, feat_idx])
            })

df_icc = pd.DataFrame(rows)

print("\nPer-feature ICC(3,k) absolute agreement:")
print(f'{"Idx":>3}  {"Feature":40s}  {"ICC(3,k)":>10}')

feature_icc_values = []

for feat_idx in range(NUMBER_OF_FEATURES):
    df_feat = df_icc[df_icc["feature"] == feat_idx]

    icc_table = pg.intraclass_corr(
        data=df_feat,
        targets="targets",
        raters="raters",
        ratings="score"
    )

    icc_row = icc_table[
        (icc_table["Type"] == "ICC3k") &
        (icc_table["CI95%"].notnull())
    ].iloc[0]

    icc_value = float(icc_row["ICC"])
    feature_icc_values.append(icc_value)

    fname = FEATURE_NAMES[feat_idx] if feat_idx < len(FEATURE_NAMES) else f"Feature {feat_idx+1}"
    print(f"{feat_idx:3d}  {fname:40s}  {icc_value:10.3f}")

overall_icc = float(np.mean(feature_icc_values))
print(f"\nOverall mean ICC(3,k): {overall_icc:.3f}")

# human vs llm agreement

print('\n' + '=' * 60)
print('B. HUMAN vs LLM AGREEMENT')
print('=' * 60)


print('\nB1. Spearman ρ between LLM and mean human scores')
print('(Per feature, plus overall correlation using all scores; ordinal 0–4 data)')

per_feature_r = {m: [] for m in LLM_MODELS}
overall_r = {}

human_flat = human_mean.reshape(-1)

for model in LLM_MODELS:
    mat = llm_mats[model]
    for feat_idx in range(NUMBER_OF_FEATURES):
        v_llm = mat[:, feat_idx]
        v_h = human_mean[:, feat_idx]
        r = safe_spearman(v_llm, v_h)
        per_feature_r[model].append(r)

    overall_r[model] = safe_spearman(mat.reshape(-1), human_flat)

# print table
header = f'{"Idx":>3}  {"Feature":40s}' + ''.join(f'{m:>10}' for m in LLM_MODELS)
print('\nPer-feature Spearman ρ:')
print(header)
for feat_idx in range(NUMBER_OF_FEATURES):
    fname = FEATURE_NAMES[feat_idx] if feat_idx < len(FEATURE_NAMES) else f'Feature {feat_idx + 1}'
    row = f'{feat_idx:3d}  {fname:40s}'
    for model in LLM_MODELS:
        row += f'{per_feature_r[model][feat_idx]:10.3f}'
    print(row)

row = f'{"ALL":>3}  {"OVERALL (flattened)":40s}'
for model in LLM_MODELS:
    row += f'{overall_r[model]:10.3f}'
print('\n' + row)

# reliability-corrected (attenuation-corrected) validity per LLM
print('\nB3. Reliability-corrected validity per LLM (attenuation-corrected ρ)')
corrected_validity = {}

print(f'{"Model":>10}  {"ρ_obs":>10}  {"ICC_llm":>10}  {"ICC_human":>11}  {"ρ_corrected":>13}')
for model in LLM_MODELS:
    r_obs = overall_r[model]
    icc_llm = LLM_ICC.get(model, float("nan"))
    icc_human = overall_icc
    prod = icc_llm * icc_human if not np.isnan(icc_llm) else float("nan")
    if np.isnan(prod) or prod <= 0.0:
        r_corr = float("nan")
    else:
        denom = float(np.sqrt(prod))
        if denom == 0.0:
            r_corr = float("nan")
        else:
            r_corr = r_obs / denom
    corrected_validity[model] = r_corr
    print(f'{model:>10}  {r_obs:10.3f}  {icc_llm:10.3f}  {icc_human:11.3f}  {r_corr:13.3f}')

# mean deviation per llm

print('\nB2. Mean absolute deviation |LLM - human mean|')
print('(Per feature, plus overall across all files × features)')

per_feature_dev = {m: [] for m in LLM_MODELS}
overall_dev = {}

for model in LLM_MODELS:
    mat = llm_mats[model]
    diffs = mat - human_mean
    per_feature_dev[model] = list(np.mean(np.abs(diffs), axis=0))
    overall_dev[model] = float(np.mean(np.abs(diffs)))

header = f'{"Idx":>3}  {"Feature":40s}' + ''.join(f'{m:>10}' for m in LLM_MODELS)
print('\nPer-feature mean absolute deviation:')
print(header)
for feat_idx in range(NUMBER_OF_FEATURES):
    fname = FEATURE_NAMES[feat_idx] if feat_idx < len(FEATURE_NAMES) else f'Feature {feat_idx + 1}'
    row = f'{feat_idx:3d}  {fname:40s}'
    for model in LLM_MODELS:
        row += f'{per_feature_dev[model][feat_idx]:10.3f}'
    print(row)

row = f'{"ALL":>3}  {"OVERALL (all scores)":40s}'
for model in LLM_MODELS:
    row += f'{overall_dev[model]:10.3f}'
print('\n' + row)

# Plot 3 — Human vs LLM Scatterplots (combined into one figure)
# One scatter per LLM (human mean vs LLM scores), arranged as:
# 2 plots top row, 2 plots middle row, 1 plot bottom row (last subplot left), last subplot hidden.
eval_dir = Path('eval_results')
eval_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(3, 2, figsize=(10, 12))
axes = axes.flatten()

x_human = human_mean.reshape(-1)

for idx, model in enumerate(LLM_MODELS):
    ax = axes[idx]
    mat = llm_mats[model]
    y_llm = mat.reshape(-1)

    grid = np.zeros((5, 5), dtype=int)
    for h, l in zip(x_human, y_llm):
        grid[int(l), int(h)] += 1

    im = ax.imshow(grid, cmap='viridis')

    ax.set_title(f'{model} (ρ = {overall_r[model]:.2f})')
    ax.set_xlabel('Mean human score')
    ax.set_ylabel('LLM score')
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_xticklabels([0, 1, 2, 3, 4])
    ax.set_yticklabels([0, 1, 2, 3, 4])

# Add colorbar in the bottom-right empty subplot
cax = axes[-1]  # bottom-right subplot you hid earlier
cax.set_visible(True)
cax.set_axis_off()  # keep the box but no ticks

fig.colorbar(im, ax=cax, fraction=0.8)

# Hide the extra (6th) subplot so layout is 2–2–1 visually
if len(axes) > len(LLM_MODELS):
    axes[-1].axis('off')

fig.tight_layout()
fig.savefig(eval_dir / 'human_vs_llm_heatmaps.png', dpi=300)
plt.close(fig)

# best llm identification

print('\n' + '=' * 60)
print('C. BEST LLM IDENTIFICATION (vs HUMAN BENCHMARKS)')
print('=' * 60)

error_var = {}
for model in LLM_MODELS:
    mat = llm_mats[model]
    diffs = mat - human_mean
    error_var[model] = float(np.var(diffs))

print('\nSummary metrics per LLM:')
print(f'{"Model":>10}  {"ρ_overall":>10}  {"Δ_overall":>10}  {"Var(error)":>12}  {"ρ_corr":>10}')
for model in LLM_MODELS:
    print(f'{model:>10}  {overall_r[model]:10.3f}  {overall_dev[model]:10.3f}  {error_var[model]:12.3f}  {corrected_validity[model]:10.3f}')

# hierarchical model selection: filter by reliability, then rank by corrected validity
reliable_models = [m for m in LLM_MODELS if LLM_ICC.get(m, 0.0) >= 0.80]

if not reliable_models:
    print('\nNOTE: No models met the ICC ≥ 0.80 reliability threshold; selecting based on all models.')
    candidate_models = LLM_MODELS
else:
    print('\nModels passing reliability filter ICC ≥ 0.80:', ', '.join(reliable_models))
    candidate_models = reliable_models

def validity_key(m):
    val = corrected_validity.get(m, float("nan"))
    return -np.inf if np.isnan(val) else val

best_model = max(candidate_models, key=validity_key)

print(f'\nChosen best-aligned LLM vs humans (hierarchical selection): {best_model}')
print('Justification:')
print(f'  Intra-LLM ICC(3,k):               {LLM_ICC[best_model]:.3f}')
print(f'  Overall Spearman ρ vs humans:     {overall_r[best_model]:.3f}')
print(f'  Reliability-corrected validity ρ: {corrected_validity[best_model]:.3f}')
print(f'  Overall mean absolute deviation Δ:{overall_dev[best_model]:.3f}')
print(f'  Variance of error:                {error_var[best_model]:.3f}')
