# Technical Resolution and Experimental Reproduction Summary

This directory contains the files and reports addressing the three outstanding issues before updating the FISAT 2026 paper:

---

## 1. Key Outcomes & Resolution Decisions

### A. Resolution of the EER Discrepancy at $k=128$
- **Findings**: The true EER for Gabor + XPCA (Residual, $\eta=0.2$) is **8.596207615815459%** (rounds to **8.60%**).
- **Explanation**: The value **8.4962%** (or **8.49%** in the LaTeX draft) is a **typographical error** and has no empirical support in the results tables or code. It likely originated from a typing slip or confusion with the cumulative explained variance at $k=128$, which is exactly **78.49%**.
- **Impact**: When the typo is corrected, XPCA is shown to **degrade** the baseline Gabor + PCA EER ($8.5178\%$) under standard settings, rather than improving it.
- **Reference**: Detailed evidence is documented in [EER_84962_VS_85962_REPORT.md](EER_84962_VS_85962_REPORT.md) and mapped in [EER_OCCURRENCES.csv](EER_OCCURRENCES.csv).

### B. Authoritative Benchmark Decision & Test-Set Leakage
- **Findings**: Both `comprehensive_benchmark.csv` and `dimension_specific_sota_results.csv` are **test-set tuned** (leaked) because their hyperparameters were optimized directly to minimize the IITD test set EER.
- **Decision**: Neither file is scientifically authoritative. We have compiled a new unbiased benchmark in [authoritative_benchmark.csv](authoritative_benchmark.csv) using fixed standard parameters (no test tuning).
- **Reference**: Technical details are in [AUTHORITATIVE_SOURCE_DECISION.md](AUTHORITATIVE_SOURCE_DECISION.md), [BENCHMARK_PROVENANCE.csv](BENCHMARK_PROVENANCE.csv), and [test_leakage_assessment.md](test_leakage_assessment.md).

### C. Paired Statistical Comparison & Generalization
- **Findings**: We ran **30 randomized split trials** comparing Gabor + PCA and Gabor + XPCA under identical conditions.
  - Standard Config EER: Baseline **7.6334%** vs. XPCA **7.8335%** ($p = 0.000005$, Wilcoxon). XPCA degrades baseline in 27/30 trials.
  - Optimized Config EER: Baseline **7.0166%** vs. XPCA **7.0524%** ($p = 0.0248$, Wilcoxon). XPCA degrades baseline in 23/30 trials.
- **Scientific Conclusion**: XPCA's scaling weights are overfitted to the training class-means (which only have 3 samples per class) and fail to generalize to unseen probe images, resulting in statistically significant performance degradation.
- **Reference**: Statistical analysis is in [PAIRED_STATISTICAL_REPORT.md](PAIRED_STATISTICAL_REPORT.md), and trials data is saved in [paired_trials.csv](paired_trials.csv) and [PAIRED_STATISTICAL_SUMMARY.csv](PAIRED_STATISTICAL_SUMMARY.csv).

---

## 2. Directory Tree of Generated Outputs

All files have been written directly to `_paper_resolution/`:

- `EER_84962_VS_85962_REPORT.md`: Detailed audit of EER values.
- `EER_OCCURRENCES.csv`: Index of where EER numbers appear in the codebase.
- `AUTHORITATIVE_SOURCE_DECISION.md`: Analysis and decision on benchmark files.
- `BENCHMARK_PROVENANCE.csv`: Mapping of benchmark script origins and leakage status.
- `authoritative_benchmark.csv`: Unbiased benchmark results.
- `PAIRED_STATISTICAL_REPORT.md`: Detailed description of statistical tests and findings.
- `paired_trials.csv`: Raw trial-by-trial results.
- `PAIRED_STATISTICAL_SUMMARY.csv`: Summarized statistics for both configurations.
- `test_leakage_assessment.md`: Technical code audit demonstrating test-set leakage.
- `CLAIM_STATUS_AFTER_RESOLUTION.csv`: Evaluation of paper claims.
- `paper_update_recommendations.md`: Practical LaTeX diffs for correcting `XPCA-1.tex`.
- `commands_executed.txt`: Shell commands used in this session.
- `environment_snapshot.txt`: Package versions and platform details.
- `configs/`: JSON files defining evaluated hyperparameters.
- `splits/`: CSV files recording random partition filenames for reproducibility.
- `raw_scores/`: Saved numpy `.npz` files of matching similarity scores for Trial 0.
- `logs/`: Command execution stdout/stderr logs.
