# Explainable PCA Framework for Palmprint Recognition

This repository implements a multi-level explainability study for PCA-based (EigenPalms) palmprint verification on the **IIT Delhi (IITD) Touchless Palmprint Database**.

---

## 1. Project Directory Structure
The workspace is organized into a modular, clean structure designed for biometric evaluation and explainability analysis:

```
├── data/
│   └── IITD/               # Windows Directory Junction to "IITD Dataset"
├── src/
│   ├── preprocessing/      # Grayscale conversion, resizing, and individual zero-mean normalization
│   ├── feature_extraction/ # PCA model fitting and Whitened/Mahalanobis projections
│   ├── evaluation/         # Cosine template matching, EER, FAR, FRR, AUC, and t-tests
│   ├── explainability/     # EigenPalm visualization, progressive reconstruction, score KDEs, and PDF generation
│   └── visualization/      # Publication-quality figure plotting
├── experiments/
│   └── run_experiments.py  # Main orchestration script running all 7 experiments and trials
├── results/
│   ├── figures/            # Standalone 300 DPI publication figures (Figures 1-8)
│   ├── tables/             # Performance tables, reconstruction stats, and t-test summaries
│   └── explainability_report.pdf # Compiled verification explainability audit report
├── paper/
│   └── findings.md         # Full academic paper draft / research conclusions
└── README.md               # Reproduction and instruction manual
```

---

## 2. Dependencies & Installation
To run the experiments, ensure you have Python 3.8+ installed along with the following packages:
```bash
pip install numpy pandas opencv-python scikit-learn scikit-image scipy matplotlib
```

---

## 3. How to Run the Experiments
The entire experimental suite, statistical validation, and report generation is fully automated:

```bash
python experiments/run_experiments.py
```

Running this command will:
1. **Load data:** Load IITD Segmented Left and Right palmprints (460 unique palm classes), resize to $128 \times 128$, and apply individual zero-mean normalization.
2. **PCA Dimension Analysis (Exp 1):** Evaluate accuracy, EER, AUC, FAR, and FRR for $k \in \{16, 32, 64, 128, 256, 512\}$. Save metrics as a CSV and plot curves.
3. **Variance Retention (Exp 2):** Save cumulative variance ratios and plot the explained variance curve highlighting elbow and saturation points.
4. **EigenPalms (Exp 3):** Reshape eigenvectors PC1-PC5, PC10, and PC20 back to image space, and render colorbar-shaded maps.
5. **Progressive Reconstruction (Exp 4):** Reconstruct representative palmprints at varying $k$ dimensions and output structural MSE, PSNR, and SSIM metrics.
6. **Scree & PC Importance (Exp 5):** Compute and plot individual eigenvalue energy distributions.
7. **Score Distribution KDE (Exp 6):** Compute KDEs for genuine and imposter match similarity scores, shading the overlap area and calculating separation distances ($d'$, Bhattacharyya).
8. **Explainability Score (ES) & PDF Compilation (Exp 7):** Compute composite ES values and compile a multi-page PDF report (`results/explainability_report.pdf`).
9. **Statistical Significance Test:** Perform **5 repeated runs** of the pipeline with randomized splits, compute EER standard deviations, and run paired t-tests comparing EER performance of $64 \text{ vs } 128$, $128 \text{ vs } 256$, and $256 \text{ vs } 512$ components.
