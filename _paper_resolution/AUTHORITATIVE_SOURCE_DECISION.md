# Decision Report: Authoritative Benchmark Source & Test-Set Leakage Analysis

This document determines the authoritative source of verification results for the FISAT 2026 paper, analyzing `comprehensive_benchmark.csv` and `dimension_specific_sota_results.csv`.

---

## 1. Grid Comparison of Benchmark Files

Both files present the performance of Gabor + PCA and Gabor + XPCA at dimensions $k \in \{32, 64, 128, 256, 512\}$. Below is their side-by-side comparison for Gabor + XPCA EER (%):

| Subspace Dimension ($k$) | comprehensive_benchmark.csv EER (%) | dimension_specific_sota_results.csv EER (%) | Parameter Source & Sweep Method |
| :---: | :---: | :---: | :--- |
| **32** | `9.5836%` | `9.5577%` | Swept per-dimension vs. fixed at $k=128$ |
| **64** | `8.6761%` | `8.5043%` | Swept per-dimension vs. fixed at $k=128$ |
| **128** | `7.9920%` | `7.9538%` | Swept per-dimension vs. fixed at $k=128$ |
| **256** | `7.7879%` | `7.6063%` | Swept per-dimension vs. fixed at $k=128$ |
| **512** | `7.1380%` | `6.9796%` | Swept per-dimension vs. fixed at $k=128$ |

---

## 2. Test-Set Leakage Audit & Provenance

We analyzed the scripts that generated these CSV files:
1. `experiments/optimize_gabor_xpca.py` (which wrote `comprehensive_benchmark.csv` / `gabor_xpca_optimized_results.csv`):
   - Grid searched Gabor parameters (orientations, lambda, sigma, downsample) and XPCA parameters (shrinkage, weights, eta) at $k=128$.
   - **Tuning protocol**: Evaluated each parameter configuration directly on the **IITD test set EER** (`y_test`, `te_scaled`).
   - Selected the "optimal" parameters that gave the minimum test EER at $k=128$ (shrinkage=0.7, eta=0.1, weights=Fisher & Stability), and applied them across all other dimensions.
2. `experiments/optimize_dimension_specific.py` (which wrote `dimension_specific_sota_results.csv`):
   - Performed an independent grid search for *each* dimension $k \in \{32, 64, 128, 256, 512\}$.
   - **Tuning protocol**: Selected the best parameter set for each dimension based on the **IITD test set EER**.
   - For example, at $k=128$, it selected `shrinkage=0.1`, `weights=Fisher Only`, and `eta=0.02`, achieving `7.9538%`.

### Scientific Conclusion:
- **Neither file is scientifically authoritative**. Both files rely on **test-set tuning** (selecting hyperparameters that minimize the test set error).
- Choosing hyperparameters on the test set leads to optimistic bias (test-set leakage) and does not reflect true generalization performance.
- When evaluated under an unbiased protocol (randomized splits without test-set tuning), the Gabor + XPCA calibration actually **degrades** the Gabor + PCA baseline (as shown in `paired_trials.csv`).

---

## 3. Decision & Recommended Corrective Actions

1. **Reject Both Legacy Tables**: For the final camera-ready paper, both `comprehensive_benchmark.csv` and `dimension_specific_sota_results.csv` must be rejected due to test-set leakage.
2. **Adopt the Authoritative Benchmark**: We have generated an unbiased benchmark in `_paper_resolution/authoritative_benchmark.csv` using the standard, fixed hyperparameters (derived prior to test-set search).
3. **Transparent Reporting**: If the leaked results are shown, they must be labeled as "Grid-searched on Test Set (UpperBound / Leaked)" to maintain scientific integrity.
