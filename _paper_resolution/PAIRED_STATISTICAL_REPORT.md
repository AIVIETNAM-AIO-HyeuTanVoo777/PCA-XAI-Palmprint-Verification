# Report: Paired Statistical Comparison & Generalization Assessment

This report provides the paired statistical comparison between **Gabor + PCA** and **Gabor + XPCA (Residual)** under identical splits, seeds, and preprocessing protocols.

---

## 1. Evaluation Protocol: Randomized Split Trials

To avoid the bias of a single deterministic split, we conducted **30 randomized split trials** (seeds 0 to 29) for two key configurations at $k=128$:
1. **Standard Configuration**: 4 orientations, $\lambda=8$, $\sigma=4$, downsample $64 \times 64$, $\eta=0.2$, weights = Standard $(0.2, 0.6, 0.2)$, shrinkage $= 0.1$.
2. **Optimized Configuration**: 6 orientations, $\lambda=10$, $\sigma=5$, downsample $52 \times 52$, $\eta=0.02$, weights = Fisher Only $(0, 1, 0)$, shrinkage $= 0.1$.

In each trial, the images of each class were randomly partitioned: 3 images for training/enrollment, and the remaining images for testing/probes. Both baseline and XPCA used the exact same split, features, and PCA projection matrix within each trial.

---

## 2. Statistical Test Results

The results of the 30 trials are summarized below:

| Statistical Metric | Standard Configuration ($k=128$) | Optimized Configuration ($k=128$) |
| :--- | :---: | :---: |
| **Mean Baseline EER** | `7.6334%` | `7.0166%` |
| **Mean XPCA EER** | `7.8335%` | `7.0524%` |
| **Mean Delta EER (XPCA - Base)** | **`+0.2002%`** (Degradation) | **`+0.0358%`** (Degradation) |
| **Median Delta EER** | `+0.2429%` | `+0.0495%` |
| **95% Confidence Interval (Delta)** | `(0.1332%, 0.2671%)` | `(0.0051%, 0.0664%)` |
| **Bootstrap 95% CI (Delta)** | `(0.1405%, 0.2650%)` | `(0.0048%, 0.0650%)` |
| **Win / Loss / Tie (Wins for XPCA)** | **3 Wins / 27 Losses** (Win Rate: 10.0%) | **7 Wins / 23 Losses** (Win Rate: 23.3%) |
| **Effect Size (Cohen's d)** | **`1.1167`** (Very Large) | **`0.4361`** (Medium) |
| **Shapiro-Wilk Test p-value** | `0.104370` (Normal) | `0.036841` (Non-normal) |
| **Paired t-test p-value** | **`0.000001`** (Highly Significant) | **`0.023643`** (Significant) |
| **Wilcoxon Signed-Rank p-value** | **`0.000005`** (Highly Significant) | **`0.024786`** (Significant) |
| **Permutation Test p-value** | **`0.000000`** (Highly Significant) | **`0.026080`** (Significant) |
| **Mean Baseline AUC** | `0.971559` | `0.975583` |
| **Mean XPCA AUC** | `0.970681` | `0.975509` |
| **Mean Delta AUC** | `-0.000878` | `-0.000075` |

---

## 3. Core Scientific Discoveries

### A. XPCA Consistently Degrades Baseline Performance
- Under the **Standard Configuration**, Gabor + XPCA (Residual) degraded the baseline in **27 out of 30 trials**. The mean EER increased by **$0.2002\%$** (p-value $< 10^{-5}$ for t-test, Wilcoxon, and permutation tests), with a very large effect size (Cohen's $d = 1.117$).
- Under the **Optimized Configuration**, Gabor + XPCA degraded the baseline in **23 out of 30 trials**. The mean EER increased by **$0.0358\%$** (p-value $\approx 0.024$), with a medium effect size (Cohen's $d = 0.436$).
- In both cases, the 95% Confidence Interval for the difference in EER is entirely above zero, indicating a statistically significant performance degradation.

### B. Explaining the Performance Degradation (Generalization Failure)
XPCA attempts to scale PCA dimensions based on class-level metrics computed on the training set:
1. **Fisher score (D)**: Measures class separability on training projections.
2. **Bootstrap instability (N)**: Measures the sensitivity of Fisher scores to training sample resamples.

However, since each class in IITD only has **3 training samples**, the class means and variances computed on these samples are highly overfitted. When we apply these scaling weights to the test (probe) samples:
- The alignment between probe features and gallery templates is perturbed by training-specific noise.
- The scaling shifts the relative distances of eigenvectors in a way that does not generalize to unseen images, resulting in higher EER and lower AUC.

---

## 4. The Source of the Legacy Success Claims
If XPCA degrades performance, why did the previous scripts report improvements?
- In `optimize_gabor_xpca.py` and `optimize_dimension_specific.py`, the hyperparameters ($\eta$, shrinkage, and weights) were selected by sweeping a grid and finding the combination that yielded the **absolute lowest EER on the test set**.
- For example, in the deterministic split, the baseline EER was `8.0095%`. By grid-sweeping on the test set, the script found that setting `shrinkage=0.1, weights=Fisher Only, eta=0.02` gave a test EER of `7.9538%` (an artificial "improvement" of `0.0558%`).
- This is a direct consequence of **test-set tuning/leakage**. When those exact same hyperparameters are evaluated on independent random splits without re-tuning on the test set, XPCA fails to generalize and degrades baseline performance.
