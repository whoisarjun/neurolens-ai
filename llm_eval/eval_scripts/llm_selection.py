"""
HOW WE SELECTED OUR LLM OF CHOICE

This script selects a single LLM as the primary scoring engine using a
three-stage, psychometrically-motivated procedure:

1. Intra-LLM reliability filter:
   - Exclude models with ICC(3,k) < RELIABILITY_THRESHOLD (default 0.80),
     a conventional cutoff for "good" reliability.

2. Inter-LLM agreement outlier filter:
   - Compute the mean pairwise inter-LLM ICC(2,1) for each model.
   - Compute the mean and standard deviation of these means.
   - Flag models whose mean inter-LLM ICC is more than OUTLIER_SD
     standard deviations below the group mean as systematic outliers,
     and exclude them (default OUTLIER_SD = 1.0).

3. Human-aligned validity ranking:
   - Among the remaining models, choose the LLM with the highest
     reliability-corrected validity (attenuation-corrected Spearman rho)
     relative to the human benchmark.
   - If there is a tie in corrected validity (within a small epsilon),
     break ties lexicographically using:
       (i) lower overall mean absolute deviation Δ_overall
      (ii) lower error variance Var(error).

Use the other files in this folder to generate values
"""

from typing import Dict, Any, List, Tuple

LLM_METRICS: Dict[str, Dict[str, float]] = {
    "deepseek": {
        "icc_intra": 0.9916343,
        "inter_icc_mean": 0.542,
        "rho_obs": 0.377,
        "rho_corrected": 0.429,
        "delta_overall": 1.030,
        "error_var": 1.352,
    },
    "granite": {
        "icc_intra": 0.9981859,
        "inter_icc_mean": 0.590,
        "rho_obs": 0.466,
        "rho_corrected": 0.528,
        "delta_overall": 0.908,
        "error_var": 1.242,
    },
    "llama": {
        "icc_intra": 0.9963090,
        "inter_icc_mean": 0.578,
        "rho_obs": 0.400,
        "rho_corrected": 0.455,
        "delta_overall": 0.910,
        "error_var": 1.217,
    },
    "ministral": {
        "icc_intra": 0.9961227,
        "inter_icc_mean": 0.556,
        "rho_obs": 0.468,
        "rho_corrected": 0.532,
        "delta_overall": 1.257,
        "error_var": 1.448,
    },
    "qwen": {
        "icc_intra": 0.9967810,
        "inter_icc_mean": 0.602,
        "rho_obs": 0.448,
        "rho_corrected": 0.508,
        "delta_overall": 0.944,
        "error_var": 1.432,
    },
}

# selection params

RELIABILITY_THRESHOLD: float = 0.80
OUTLIER_SD: float = 1.0
EPS: float = 1e-3

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")

def _std(xs: List[float]) -> float:
    m = _mean(xs)
    var = _mean([(x - m) ** 2 for x in xs]) if xs else float("nan")
    return var ** 0.5

# selection pipeline

# step 1
def filter_by_reliability(metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    return {
        name: m
        for name, m in metrics.items()
        if m.get("icc_intra", 0.0) >= RELIABILITY_THRESHOLD
    }

# step 2
def filter_outliers_by_inter_icc(metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    inter_means: List[float] = [m["inter_icc_mean"] for m in metrics.values()]
    mu = _mean(inter_means)
    sigma = _std(inter_means)

    if sigma == 0 or sigma != sigma:
        return metrics

    threshold = mu - OUTLIER_SD * sigma

    filtered = {
        name: m
        for name, m in metrics.items()
        if m["inter_icc_mean"] >= threshold
    }
    return filtered

# stage 3
def rank_candidates(metrics: Dict[str, Dict[str, float]]) -> Tuple[str, Dict[str, Any]]:
    def sort_key(item: Tuple[str, Dict[str, float]]) -> Tuple[float, float, float]:
        name, m = item
        rho_corr = m["rho_corrected"]
        delta = m["delta_overall"]
        var_err = m["error_var"]
        return (-rho_corr, delta, var_err)

    sorted_items = sorted(metrics.items(), key=sort_key)
    best_name, best_metrics = sorted_items[0]
    return best_name, best_metrics

# main
def main() -> None:
    print("=" * 60)
    print("LLM SELECTION BASED ON RELIABILITY, AGREEMENT, AND VALIDITY")
    print("=" * 60)

    print("\nINPUT METRICS PER MODEL:")
    print(f'{"Model":>10}  {"ICC_intra":>10}  {"ICC_inter_mean":>15}  '
          f'{"ρ_obs":>8}  {"ρ_corr":>8}  {"Δ_overall":>10}  {"Var(error)":>12}')
    for name, m in LLM_METRICS.items():
        print(f'{name:>10}  {m["icc_intra"]:10.3f}  {m["inter_icc_mean"]:15.3f}  '
              f'{m["rho_obs"]:8.3f}  {m["rho_corrected"]:8.3f}  '
              f'{m["delta_overall"]:10.3f}  {m["error_var"]:12.3f}')

    reliable = filter_by_reliability(LLM_METRICS)
    print("\nStage 1 – Reliability filter (ICC_intra ≥ "
          f"{RELIABILITY_THRESHOLD:.2f}):")
    print("Models retained:", ", ".join(sorted(reliable.keys())))

    after_inter = filter_outliers_by_inter_icc(reliable)
    print("\nStage 2 – Inter-LLM agreement outlier filter "
          f"(mean ICC(2,1) < μ - {OUTLIER_SD:.1f}·σ removed):")
    print("Models retained:", ", ".join(sorted(after_inter.keys())))

    best_name, best_metrics = rank_candidates(after_inter)

    print("\nStage 3 – Human-aligned validity ranking:")
    print(f"Chosen LLM: {best_name}")
    print("\nJustification for chosen LLM:")
    print(f"  Intra-LLM ICC(3,k):                 {best_metrics['icc_intra']:.3f}")
    print(f"  Mean inter-LLM ICC(2,1):            {best_metrics['inter_icc_mean']:.3f}")
    print(f"  Overall Spearman ρ vs humans:       {best_metrics['rho_obs']:.3f}")
    print(f"  Reliability-corrected validity ρ:   {best_metrics['rho_corrected']:.3f}")
    print(f"  Overall mean absolute deviation Δ:  {best_metrics['delta_overall']:.3f}")
    print(f"  Variance of error:                  {best_metrics['error_var']:.3f}")

if __name__ == "__main__":
    main()
