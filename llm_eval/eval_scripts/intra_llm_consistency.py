from pathlib import Path
import json
from collections import Counter
from features import semantics
import pandas as pd
import numpy as np

files = ['deepseek', 'granite', 'llama', 'ministral', 'qwen']

print('\n' + '=' * 60)
print('INTRA-LLM CONSISTENCY')
print('=' * 60)

NUMBER_OF_RUNS = 5
NUMBER_OF_FEATURES = 18

def icc_3k(data: np.ndarray) -> float:
    if data.ndim != 2:
        raise ValueError("data must be 2D (n_targets, n_raters)")

    n, k = data.shape
    if n < 2 or k < 2:
        return float("nan")

    mean_rows = data.mean(axis=1, keepdims=True)
    mean_cols = data.mean(axis=0, keepdims=True)
    grand_mean = data.mean()

    msr = k * np.sum((mean_rows - grand_mean) ** 2) / (n - 1)
    mse = np.sum((data - mean_rows - mean_cols + grand_mean) ** 2) / ((n - 1) * (k - 1))

    if msr == 0:
        return float("nan")

    return (msr - mse) / msr

data = {}

icc_by_model = {}
mean_icc_by_model = {}

identical_scores = {}
pm1_scores = {}
avg_variances = {}

for f in files:
    with Path(f'models/features/{f}.json').open() as j:
        results = [r.get('results') for r in json.load(j).get('results')]
        identical = 0.0
        pm_1 = 0.0
        sum_var = [0.0 for _ in range(NUMBER_OF_FEATURES)]
        feature_mats = [[] for _ in range(NUMBER_OF_FEATURES)]

        for r in results:
            tuples = list(map(tuple, r))
            counts = Counter(tuples)
            most_common_count = max(counts.values())
            identical += most_common_count / len(r)

            stable = 0
            for i in range(NUMBER_OF_FEATURES):
                features = [s[i] for s in r]
                var = max(features) - min(features)
                if var <= 1:
                    stable += 1
                sum_var[i] += var

                feature_mats[i].append(features)

            pm_1 += stable / NUMBER_OF_FEATURES

        variance = [v / len(results) for v in sum_var]

        icc_per_feature = []
        for i in range(NUMBER_OF_FEATURES):
            mat = np.array(feature_mats[i], dtype=float)
            if mat.size == 0 or np.all(mat == mat.flat[0]):
                icc_val = float("nan")
            else:
                icc_val = icc_3k(mat)
            icc_per_feature.append(icc_val)

        mean_icc = float(np.nanmean(icc_per_feature))

    print(f'\nLLM model: {f}')

    identical_pcn = identical / len(results)
    print(f'% Identical outputs across all transcripts: {(100 * identical_pcn):.1f}%')

    pm_1_pcn = pm_1 / len(results)
    print(f'% ±1 Deviation across all transcripts: {(100 * pm_1_pcn):.1f}%')

    avg_variance = sum(variance) / len(variance)
    print(f'Average feature variance: {avg_variance:.3f}')

    icc_by_model[f] = icc_per_feature
    mean_icc_by_model[f] = mean_icc

    print(f'Overall mean ICC(3,k): {mean_icc:.7f}')

    identical_scores[f] = identical_pcn
    pm1_scores[f] = pm_1_pcn
    avg_variances[f] = avg_variance

    data[f] = variance

rows = []
for idx, feat_name in enumerate(semantics.all_features):
    row = {
        'feature_index': idx,
        'feature_name': feat_name,
    }
    for m in files:
        row[f'{m}_variance'] = data[m][idx]
        row[f'{m}_icc_3k'] = icc_by_model[m][idx]
    rows.append(row)

df = pd.DataFrame(rows)

eval_dir = Path('eval_results')
eval_dir.mkdir(parents=True, exist_ok=True)
csv_path = eval_dir / 'feature_variance.csv'
df.to_csv(csv_path, index=False)

print("\nAll feature variance and ICC(3,k) values saved to eval_results/feature_variance.csv")

print("\nModel ranking by mean ICC(3,k):")
ranked = sorted(mean_icc_by_model.items(), key=lambda kv: kv[1], reverse=True)
for rank, (model_name, icc_mean) in enumerate(ranked, start=1):
    print(f" {rank}. {model_name}: {icc_mean:.7f}")
