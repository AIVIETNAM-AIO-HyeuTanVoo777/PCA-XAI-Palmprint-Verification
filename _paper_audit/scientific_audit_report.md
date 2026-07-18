# Master Technical & Scientific Audit Report

This report presents a comprehensive technical and scientific audit of the palmprint verification research project, focusing on Principal Component Analysis (PCA) feature compression, Gabor preprocessing, utility-weighted subspace metrics, and the mathematical risk of coordinate collapse. The objective is to compile and document all necessary scientific, dataset, and code-level findings to facilitate the completion and submission of the manuscript to the **FISAT 2026** conference under the **Springer LNCS** format.

---

## 1. Executive Summary
The audited project examines two core avenues in lightweight palmprint recognition:
1. **Explainable PCA Auditing:** Analyzing the representation space of EigenPalms across different dimensions $k \in [16, 512]$ to establish the relationship between cumulative variance, visual reconstruction fidelity (SSIM, PSNR), and verification metrics (EER, AUC).
2. **Coordinate Collapse & Subspace Metric Learning:** Diagnosing why non-uniform diagonal scaling (naive weighting based on components' biometric utility) degrades matching performance under angular metrics, and validating a mathematically bounded **Residual Calibration** mitigation.

Key outcomes of the audit include:
- **EER Resolution:** Clarified and mapped the 5 different EER values obtained at $k=128$ (8.51%, 10.73%, 8.49% for standard Gabor vs. 8.01% and 7.95% for optimized configurations).
- **Dataset Auditing:** Confirmed hand consistency and verified the 18.3% cross-session identity swap anomaly in the Tongji dataset, which is resolved via Hungarian assignment remapping.
- **Mathematical Validation:** Reviewed and formalized the condition number proofs for coordinate collapse and bounded residual calibration.
- **LaTeX Compliance:** Identified template discrepancies (currently IEEEtran) and provided a transition guide to the target Springer LNCS layout.

---

## 2. Directory Architecture & Data Loading Protocols
The project root directory is structured as follows:
- `src/`: Core implementation files.
  - `preprocessing/`: Gabor filter bank (`gabor.py`) and dataset loading (`loader.py`).
  - `feature_extraction/`: PCA and subspace projections.
  - `explainability/`: Metric calculation, component ablation, and CAES++ calculation (`xpca.py`).
  - `evaluation/`: EER and ROC evaluation modules (`metrics.py`).
- `experiments/`: Orchestration and parameter optimization scripts.
  - `run_experiments.py`: Baseline PCA experiments on IITD.
  - `run_gabor_xpca_experiments.py`: Baseline vs. standard XPCA calibration.
  - `optimize_gabor_xpca.py`: Hyperparameter sweep.
  - `optimize_dimension_specific.py`: Dimension-specific grid search yielding SOTA results.
- `results/`: CSV output tables and plot graphics.
- `paper/`: Main explainable PCA LaTeX manuscript and supporting documents.
- `XPCA-1.tex`: Manuscript draft for the coordinate collapse study.

### Verification Splits & Protocols
- **IITD Dataset:** 230 subjects, left/right hands separated $\rightarrow$ 460 unique classes, 2,601 images.
  - *Split protocol:* First 3 sorted images per class form the gallery/training set ($N_{train}=1,380$). Remaining images are probes ($N_{test}=1,221$).
  - *Evaluation:* 1,221 genuine comparisons, 560,439 imposter comparisons.
- **Tongji Dataset:** 600 unique classes, 12,000 images. Evaluated on a 100-subject subset.
  - *Split protocol:* Session 1 for training/gallery (1,000 images), Session 2 for testing/probes (1,000 images).
  - *Evaluation:* 1,000 genuine comparisons, 99,000 imposter comparisons.

---

## 3. Dataset Anomaly Auditing
- **IITD Dataset:** No anomalies detected. Subject IDs in filenames match the data loading indices perfectly.
- **Tongji Dataset:** Programmatic analysis using the Hungarian optimal assignment algorithm revealed that **110 out of 600 subjects (18.3%) are misaligned** between Session 1 and Session 2. Without remapping, this misalignment inflates False Rejects and degrades EER to ~34%. Using the mapping in `Tongji_session2_mapping.csv` aligns the identities, restoring EER to $<1\%$.

Detailed reports are available in:
- [Hand Consistency & Separation Audit](file:///d:/0.Research/AILLLL/PCA-XAI-Palm%20-%20Copy%20%282%29/_paper_audit/hand_consistency_check.md)
- [Identity Mapping Consistency Audit](file:///d:/0.Research/AILLLL/PCA-XAI-Palm%20-%20Copy%20%282%29/_paper_audit/identity_mapping_check.md)
- [Session Split & Protocol Verification](file:///d:/0.Research/AILLLL/PCA-XAI-Palm%20-%20Copy%20%282%29/_paper_audit/session_protocol_check.md)

---

## 4. Mathematical Modeling: Coordinate Collapse & Residual Calibration
Diagonal metric learning scales feature coordinates by their utility: $z'_i = a_i z_i$. However, under cosine matching, the diagonal weights are squared, yielding the effective metric tensor $M = A^2$.

- **Coordinate Collapse (Theorem 1):** Applying min-max normalized utility weights ($a_i \in [\epsilon, 1]$) causes the condition number $\kappa(M) = 1/\epsilon^2$ to diverge as $\epsilon \to 0$. The metric tensor becomes ill-conditioned, and matching is dominated by the single component with the highest utility, collapsing the functional dimensionality to 1D and degrading verification accuracy.
- **Mitigation (Theorem 2):** Bounded **Residual Calibration** ($a_i = 1 + \eta \cdot S_i^{\text{norm}}$) restricts the condition number of the metric tensor:
  $$\kappa(M) \leq (1 + \eta)^2$$
  Setting $\eta \leq 0.15$ guarantees $\kappa(M) \leq 1.32$, preventing collapse while introducing a controlled bias toward high-utility features.

Formal proofs are available in:
- [Coordinate Collapse Theory & Metric Conditioning Proofs](file:///d:/0.Research/AILLLL/PCA-XAI-Palm%20-%20Copy%20%282%29/_paper_audit/coordinate_collapse_theory.md)
- [Verification Pipeline & Metric Validation](file:///d:/0.Research/AILLLL/PCA-XAI-Palm%20-%20Copy%20%282%29/_paper_audit/metric_validation.md)

---

## 5. Experimental Results & Verification Benchmarks
The experimental tables have been compiled in the audit directory:
- [PCA Dimension Analysis](file:///d:/0.Research/AILLLL/PCA-XAI-Palm%20-%20Copy%20%282%29/_paper_audit/pca_dimension_analysis.csv)
- [Closed-Set Repeated-Trial Statistics](file:///d:/0.Research/AILLLL/PCA-XAI-Palm%20-%20Copy%20%282%29/_paper_audit/statistical_summary.csv)
- [Closed-Set Repeated-Trial Significance (t-test)](file:///d:/0.Research/AILLLL/PCA-XAI-Palm%20-%20Copy%20%282%29/_paper_audit/statistical_significance.csv)
- [Open-Set Repeated-Trial Statistics](file:///d:/0.Research/AILLLL/PCA-XAI-Palm%20-%20Copy%20%282%29/_paper_audit/open_set_statistical_summary.csv)
- [Open-Set Repeated-Trial Significance (t-test)](file:///d:/0.Research/AILLLL/PCA-XAI-Palm%20-%20Copy%20%282%29/_paper_audit/open_set_statistical_significance.csv)
- [CAES++ Scores (empirical Information Bottleneck)](file:///d:/0.Research/AILLLL/PCA-XAI-Palm%20-%20Copy%20%282%29/_paper_audit/caes_scores.csv)
- [Compression & Description Length Trade-offs](file:///d:/0.Research/AILLLL/PCA-XAI-Palm%20-%20Copy%20%282%29/_paper_audit/compression_tradeoff.csv)
- [Gabor + XPCA Calibration Comparison (k=128)](file:///d:/0.Research/AILLLL/PCA-XAI-Palm%20-%20Copy%20%282%29/_paper_audit/gabor_xpca_comparison.csv)
- [Dimension-Specific SOTA XPCA Configurations](file:///d:/0.Research/AILLLL/PCA-XAI-Palm%20-%20Copy%20%282%29/_paper_audit/dimension_specific_sota_results.csv)

### Resolution of EER Discrepancies at k=128
The different EER values at $k=128$ correspond to standard versus optimized Gabor/XPCA parameter setups. Under standard Gabor preprocessing (4 orientations), min-max scaling degrades EER to **10.73%** (collapse), while residual calibration yields **8.49%**. Optimizing the Gabor filters (6 orientations) improves the baseline Gabor-PCA to **8.01%**, and dimension-specific residual tuning yields the optimal EER of **7.95%**.
See details: [Resolution of EER Discrepancies at Dimension k=128](file:///d:/0.Research/AILLLL/PCA-XAI-Palm%20-%20Copy%20%282%29/_paper_audit/dim_128_eer_discrepancy_resolution.md)

---

## 6. Publication Readiness & LaTeX Style Adaptation
Both draft manuscripts (`XPCA-1.tex` and `paper/main.tex`) are currently formatted in IEEEtran double-column conference style.
To submit to **FISAT 2026** (Springer LNCS format), they must be converted to single-column LNCS format.
- Major changes include replacing `\documentclass[conference]{IEEEtran}` with `\documentclass[runningheads]{llncs}`, structuring title and authors with the `\institute` macro, placing keywords inside the abstract block using the `\keywords{...}` command, removing manual theorem definition overrides, and using the `splncs04` bibliography style.
See step-by-step conversion guidelines: [LaTeX Style & Template Compliance Check](file:///d:/0.Research/AILLLL/PCA-XAI-Palm%20-%20Copy%20%282%29/_paper_audit/latex_style_compliance_check.md)

---

## 7. Next Steps & Recommendations
To compile the final paper bundle:
1. **Remerge findings:** Integrate the coordinate collapse proofs and SOTA EER (7.95%) from `XPCA-1.tex` into the main explainability paper `paper/main.tex` to form a single, high-impact publication on "Geometry-Preserving Metric Calibration and Explainability for Palmprint Subspaces."
2. **Convert to LNCS:** Execute the step-by-step conversion guide on the merged LaTeX file.
3. **Generate figures:** Compile the graphics from `results/figures/` (reconstructions, score distributions, and Gabor feature maps) into standard vector EPS/PDF formats for final inclusion.
