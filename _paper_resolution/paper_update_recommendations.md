# Revision Recommendations for Springer LNCS Manuscript (`XPCA-1.tex`)

This document outlines the required changes to the LaTeX draft `XPCA-1.tex` to correct the EER typographical errors, address the test-set leakage, and update the scientific claims.

---

## 1. Correcting the EER Typographical Error

The current text in `XPCA-1.tex` contains the typo `8.49%` for standard Residual XPCA at $k=128$, which must be corrected to the true reproducible value of `8.60%` (or `8.5962%`).

### Text Correction (Line 189)
```diff
-Conversely, our proposed Residual Calibration bounds the metric tensor, avoiding collapse and marginally improving the baseline to $8.49\%$ (prior to full optimization).
+Conversely, our proposed Residual Calibration bounds the metric tensor, preventing the catastrophic collapse of min-max scaling, though it remains close to the baseline at $8.60\%$ (prior to full optimization).
```

### Table 2 Correction (Line 202)
In the LaTeX table containing the comparative results, correct the row for standard Residual XPCA:
```diff
-Gabor + XPCA (Residual, $\eta=0.2$) [Bounded] & 8.49 & 0.9693 \\
+Gabor + XPCA (Residual, $\eta=0.2$) [Bounded] & 8.60 & 0.9641 \\
```

---

## 2. Addressing Test-Set Leakage in the SOTA Section

The reported optimized EER of `7.95%` (at $k=128$) was obtained by selecting hyperparameters directly on the test set. To maintain scientific integrity, the manuscript must disclose this tuning protocol or replace the numbers with unbiased cross-validation results.

### Text Revision (Section 4.3 / 4.4)
Add a disclosure statement regarding hyperparameter selection:
> **Note on Hyperparameter Selection**: The optimized configuration parameters (orientations=6, lambda=10, shrinkage=0.1, eta=0.02) represent the empirical upper bound of the method's potential, as they were tuned via grid search on the test set. When evaluated under a strict paired randomized-split protocol (without test-set leakage), Gabor + XPCA (Residual) yields a mean EER of $7.05\%$ compared to the Gabor + PCA baseline of $7.02\%$ ($p$-value $= 0.024$).

---

## 3. Recommended Table Update: Authoritative Benchmark

We recommend replacing the legacy Table 3 (if any) or adding a new table presenting the unbiased results from `authoritative_benchmark.csv`:

```latex
\begin{table}[h]
\centering
\caption{Unbiased Verification Results (EER \%) across Subspace Dimensions}
\begin{tabular}{lccccc}
\hline
Method & $k=32$ & $k=64$ & $k=128$ & $k=256$ & $k=512$ \\
\hline
Raw PCA Baseline & 10.65 & 9.91 & 9.43 & 9.32 & 9.25 \\
Gabor + PCA Baseline & 10.06 & 8.93 & 8.52 & 8.10 & 7.87 \\
Gabor + XPCA (Residual, $\eta=0.2$) & 10.17 & 8.99 & 8.60 & 8.29 & 8.05 \\
\hline
\end{tabular}
\end{table}
```

---

## 4. Key Explanations to Include in Discussion

1. **Catastrophic Collapse of Min-Max Scaling**: Explain that Min-Max scaling of utility weights is highly unstable because it maps the lowest-utility dimension to 1.0 and the highest to $1 + \eta$, which over-amplifies the low-variance noise components and leads to severe EER degradation (e.g. $11.47\%$ at $k=32$).
2. **Robustness of Residual Scaling**: Highlight that Residual Calibration avoids this collapse by bounding the weights around 1.0 (between $1.0$ and $1 + \eta$). This bounds the metric tensor and keeps performance close to the PCA baseline, which is a major stability benefit even if it does not outperform the baseline.
3. **Overfitting of Class Metrics**: Explain that since the training set contains only 3 samples per class, the estimated class-means and variances are highly noisy. Thus, any scaling derived from them is overfit to the training set and degrades generalization on unseen probe samples.
