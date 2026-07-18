# Code Audit: Test-Set Leakage and Tuning Protocols

This document provides a code-level audit of the parameter tuning scripts in the codebase, proving that the hyperparameters reported in the paper were optimized directly on the test set.

---

## 1. Audit of `experiments/optimize_gabor_xpca.py`

This script is responsible for finding the "optimal" Gabor parameters and the fixed XPCA parameters used in `comprehensive_benchmark.csv`.

### A. Gabor Parametric Sweep (Phase 1)
In lines 75-77:
```python
        res = evaluate_verification(tr_proj_temp, y_train, te_proj_temp, y_test, n_classes)
        eer_val = res["eer"] * 100
        print(f"--> EER for {cfg['name']} at k=128: {eer_val:.4f}%")
```
- Here, the test projections (`te_proj_temp`) and test labels (`y_test`) are passed to `evaluate_verification`.
- The best Gabor configuration is selected based on the lowest test EER value (`eer_val < best_gabor_eer`).

### B. XPCA Parameter Sweep (Phase 2)
In lines 221-225:
```python
                res = evaluate_verification(tr_scaled, y_train, te_scaled, y_test, n_classes)
                eer_val = res["eer"] * 100
                
                if eer_val < best_xpca_eer:
                    best_xpca_eer = eer_val
```
- Again, the scaled test projections (`te_scaled`) and test labels (`y_test`) are evaluated to compute the test set EER.
- The hyperparameters (`shrinkage`, `weights`, and `eta`) are selected to minimize this test set EER.

---

## 2. Audit of `experiments/optimize_dimension_specific.py`

This script generates `dimension_specific_sota_results.csv` by finding the best XPCA configuration for each dimension independently.

In lines 201-207:
```python
                    res_test = evaluate_verification(tr_opt_scaled, y_train, te_opt_scaled, y_test, n_classes)
                    eer_val = res_test["eer"] * 100
                    
                    if eer_val < best_k_eer:
                        best_k_eer = eer_val
                        best_k_params = { ... }
```
- For every dimension $k$, the script iterates over the parameter grid and evaluates performance on the test set (`te_opt_scaled` and `y_test`).
- The hyperparameters saved in the SOTA table are chosen purely because they minimize the test EER for that dimension.

---

## 3. Scientific Impact of the Leakage

- **Overestimated Performance**: Evaluating hyperparameter selection on the test set leads to a highly optimistic EER bias. The reported EER values of `7.95%` or `7.99%` are artificial upper bounds.
- **Generalization Failure**: When these "optimized" hyperparameters are evaluated on independent random splits (as shown in our paired trials report), the performance of Gabor + XPCA degrades the baseline because the parameters are overfitted to the specific characteristics of the deterministic test set.
- **Violation of Scientific Standards**: In machine learning and biometrics, hyperparameters must be tuned using cross-validation on the training set or a separate validation split. The test set must be kept completely blind until the final evaluation.

---

## 4. Recommendations for FISAT 2026 Revision

To make the paper scientifically sound, we recommend:
1. **Report Unbiased Results**: Use the standard, fixed configurations from `_paper_resolution/authoritative_benchmark.csv` where parameters are fixed a priori (eta=0.2, weights=(0.2, 0.6, 0.2)), and clearly state that XPCA does not outperform the PCA baseline.
2. **Nested Cross-Validation**: If hyperparameter tuning is required, implement a nested cross-validation protocol:
   - Partition the training set (3 images per class) into train/validation folds.
   - Tune XPCA parameters on the validation folds.
   - Evaluate the final selected parameter set on the independent test set.
3. **Reserving a Development Set**: Partition the subjects into a training cohort (e.g. 150 subjects) and testing cohort (80 subjects) so that all parameter tuning is performed on the training cohort, leaving the testing cohort untouched.
