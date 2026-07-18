# Explainable PCA Palmprint Verification: Conference Paper Draft Sections

This file contains the polished, LaTeX-friendly draft sections for the paper: **"Explainable PCA Framework for Palmprint Recognition: A Study on Feature Compression, Interpretability and Verification Performance"**.

---

## SECTION IV. RESULTS AND DISCUSSION

In this section, we present the empirical results of our multi-level explainability audit of Principal Component Analysis (PCA) on the IIT Delhi (IITD) Touchless Palmprint Database. The evaluations decompose the PCA representation space into global variance analysis, spatial-semantic activations (EigenPalms), progressive reconstruction fidelity, matching score distributions, and statistical significance tests.

### A. Global Variance and Dimension Analysis
The global representational capability of PCA is characterized by the eigenvalues corresponding to each principal component. The Scree plot exhibits a steep descent, with the first $5$ components capturing $51.2\%$ of the total variance, followed by a long-tail distribution where individual variance contributions fall below $0.01\%$ beyond $k=100$. 

The cumulative explained variance ratio shows a sharp elbow point around $k=32$, retaining $85.92\%$ of the total signal energy. It crosses the $95\%$ saturation threshold at $k=128$ ($95.34\%$), and converges to $98.71\%$ at $k=512$. This indicates that the global distribution is highly compressed, and a small subset of components encodes the vast majority of image variance.

### B. EigenPalm Semantic Mapping
Visual analysis of the reshaped eigenvectors (EigenPalms) reveals a clear hierarchical decomposition of palm structures:
* **PC 1** captures the low-frequency global illumination gradients and coarse hand boundaries. It exhibits minimal gradient or Laplacian energy, acting primarily as a bias component.
* **PC 2 to PC 4** map directly to the coarse, high-contrast anatomical landmarks of the palm, specifically the primary flexion creases (the Life Line, Head Line, and Heart Line).
* **PC 5 to PC 10** capture secondary wrinkles and medium-frequency local creases.
* **PC 20 and beyond** transition into diffuse, grid-like noise patterns, pixel-level reflections, and minor sensor boundary alignment artifacts.

### C. Progressive Reconstruction Fidelity
We evaluate the structural information lost during compression by progressively reconstructing the palmprint images across dimensions $k$. As shown in the progressive metrics, the structural similarity index (SSIM) rises from $0.5518$ at $k=16$ to $0.9234$ at $k=128$, eventually reaching $0.9854$ at $k=512$. 

At $k=16$, the reconstruction preserves only blurred hand shapes and low-frequency gradients. The primary palm lines begin to resolve between $k=32$ and $k=64$. By $k=128$, the structural details are recovered with high fidelity, and the reconstructed image becomes visually indistinguishable from the original, except for high-frequency skin textures.

### D. Match Score Distribution and Separation
To explain how verification decisions are affected in the subspace, we analyze the matching score distributions at the optimal dimension of $k=128$. The cosine similarities of genuine and imposter comparisons are modeled using Kernel Density Estimation (KDE). 

The distribution analysis yields a Separation Distance ($d'$) of $3.842$, a Bhattacharyya Distance ($D_B$) of $1.854$, and an overlap area of $0.0412$. Verification errors occur exclusively in the overlap region between similarity thresholds of $0.65$ and $0.75$. In this region, imposter comparisons involving palms with similar global geometries match the similarity scores of genuine comparisons suffering from scaling or out-of-plane rotation noise.

### E. Statistical Significance Testing
To assess if performance differences across dimensions are statistically significant, we conduct five repeated runs with randomized train/test splits. We evaluate EER values using a Relational Paired t-test:
1. **$64$ vs. $128$ components:** Yields a t-statistic of $4.2400$ and a two-tailed $p$-value of $0.0133$. Since $p < 0.05$, the EER reduction from $9.18\%$ to $8.85\%$ is statistically significant.
2. **$128$ vs. $256$ components:** Yields a t-statistic of $-0.2142$ and a $p$-value of $0.8409$. The performance difference is not statistically significant.
3. **$256$ vs. $512$ components:** Yields a t-statistic of $-0.5475$ and a $p$-value of $0.6131$. The EER degradation from $8.85\%$ to $9.00\%$ is not statistically significant.

This demonstrates that $k=128$ represents a hard statistical performance boundary for PCA on this dataset; allocating additional components beyond this threshold fails to improve verification.

---

## SECTION V. TRADE-OFF AND EXPLAINABILITY SCORE ANALYSIS

In biometric system design, dimensionality selection is a multi-criteria optimization problem. Designers must balance compression (computational efficiency), representational quality (interpretability), and verification performance (accuracy). Table I summarizes these trade-offs.

\begin{table}[htbp]
\caption{Dimensional Trade-off and Explainability Metrics}
\label{tab:tradeoffs}
\centering
\begin{tabular}{|c|c|c|c|c|c|}
\hline
\textbf{PCs ($k$)} & \textbf{\vtop{\hbox{\strut Variance}\hbox{\strut Retained (\%)}}} & \textbf{SSIM} & \textbf{\vtop{\hbox{\strut Balanced}\hbox{\strut Accuracy (\%)}}} & \textbf{EER (\%)} & \textbf{\vtop{\hbox{\strut Explainability}\hbox{\strut Score (ES)}}} \\ \hline
16  & 77.41\% & 0.5518 & 88.53\% & 11.47\% & 0.0000 \\ \hline
32  & 85.92\% & 0.7241 & 90.17\% & 9.83\%  & 0.4497 \\ \hline
64  & 91.80\% & 0.8532 & 90.82\% & 9.18\%  & 0.7512 \\ \hline
128 & 95.34\% & 0.9234 & 91.15\% & 8.85\%  & 0.9239 \\ \hline
256 & 97.45\% & 0.9587 & 91.15\% & 8.85\%  & 0.9680 \\ \hline
512 & 98.71\% & 0.9854 & 91.00\% & 9.00\%  & 1.0000 \\ \hline
\end{tabular}
\end{table}

### A. Diminishing Returns and Over-parameterization
The empirical data shows that while representational quality (variance retention and SSIM) increases continuously up to $k=512$, verification performance peaks and saturates at $k=128$. Beyond $k=128$, the marginal return on EER becomes zero, and at $k=512$, the EER slightly degrades to $9.00\%$. 

This discrepancy occurs because early components capture the coarse structural palm lines which are highly stable and discriminative. High-dimensional components ($k > 128$) capture high-frequency skin textures and minor sensor-level reflections. While these high-frequency components are necessary to visually reconstruct the image (as reflected by the high SSIM of $0.9854$ at $k=512$), they do not carry stable biometric identity signatures and instead introduce noise into the Cosine Similarity computation.

### B. Explainability Score (ES) Interpretation
The proposed Explainability Score (ES) integrates these competing criteria:
$$ES(k) = \alpha \cdot \text{VarRet}(k) + \beta \cdot \text{ReconQual}(k) + \gamma \cdot \text{VerifPerf}(k)$$
With $\alpha=0.30$, $\beta=0.30$, and $\gamma=0.40$, using min-max scaling to normalize each term to $[0, 1]$ across the evaluated dimensions.

The ES curve reaches its maximum of $1.0000$ at $k=512$. This occurs because the cumulative variance ($98.71\%$) and the reconstruction fidelity ($SSIM = 0.9854$) continue to rise, making $k=512$ the **most interpretable representation space** (i.e., the subspace where the original image structure is most fully preserved). 

However, $k=128$ represents the **practical operational boundary** ($ES = 0.9239$). At $k=128$, the verification performance achieves its maximum ($EER = 8.85\%$), the variance retention is above the $95\%$ target, and the structural reconstruction is highly resolved ($SSIM = 0.9234$). Using $k=128$ achieves a **128-fold reduction** in feature dimensions compared to the raw pixel space while avoiding the noise-fitting behaviors present at higher dimensions.

---

## SECTION VI. THREATS TO VALIDITY

To ensure scientific rigor and transparency, we identify and discuss the primary threats to the validity of this study.

### A. Internal Validity
Internal threats relate to experimental design choices and parameter settings. 
* **Preprocessing Impact:** The baseline EER is highly sensitive to our choice of preprocessing. Removing the individual sample zero-mean normalization degrades the EER at $k=128$ from $8.85\%$ to $33.38\%$, as the Cosine Similarity becomes dominated by global illumination offsets rather than biometric structures. 
* **PCA Subspace Constraints:** The eigenvectors are computed using a maximum of $512$ components due to the training set size constraints. Although $512$ components cover $98.71\%$ of the explained variance, they do not span the entire mathematical image space.

### B. Dataset Validity
* **Database Homogeneity:** All experiments are conducted exclusively on the segmented subset of the IITD dataset. While this allows us to study clean biometric signals without hand-boundary detection errors, it assumes perfect segmentation. 
* **Demographics:** The IITD database was collected in an academic environment with limited age and demographic variation, which may skew the texture-level representation.

### C. External Validity
* **Generalization to Contactless Sensors:** The IITD database is touchless but collected under relatively stable illumination and position constraints. Under unconstrained contactless scenarios (e.g., mobile phone cameras), severe translation, scale, out-of-plane rotation, and motion blur will occur. The performance of a rigid PCA projection under these variations is expected to degrade.
* **Cross-Sensor Generalization:** The PCA eigenvectors are fitted on the sensor characteristics of the IITD camera. Applying these templates to images captured by different sensors without domain adaptation would result in matching mismatches.

### D. Methodological Validity
* **Enrollment Protocol:** Our template matching protocol uses the mean training projection of the first $3$ samples. While this is a standard protocol in palmprint research, varying the number of training samples (e.g., using $1$ or $2$ samples for enrollment) would alter the template stability and shift the performance saturation point.
* **Classifier Simplicity:** We use Cosine Similarity for template matching because it is a parameter-free, computationally efficient metric that directly reflects the feature space structure. Using advanced classifiers (e.g., Support Vector Machines or neural networks) might yield lower EERs but would obscure the direct relationship between PCA dimensions and matching accuracy.

### E. Statistical Validity
* **Split Variation:** Although we utilize five repeated runs with randomized splits to compute standard deviations and t-test statistics, the number of trials ($M=5$) is modest. A larger number of runs (e.g., $M=20$) would provide tighter confidence intervals, though the current results are statistically significant ($p = 0.0133$) at the critical $64$ vs. $128$ boundary.

---

## SECTION VII. CONCLUSION

This paper presented an explainable PCA framework for touchless palmprint verification on the IITD database. By auditing the representations at multiple levels, we mapped PCA components to physical biometric structures, showing that early components ($k \le 64$) capture the primary flexion lines, while mid-to-high components capture local wrinkles and texture. 

Our trade-off analysis proved that verification performance saturates at $k=128$ (EER $8.85\%$), and that higher components capture noise rather than stable identity signatures. The proposed Explainability Score (ES) successfully modeled these trade-offs, demonstrating that while $k=512$ provides the highest visual and variance interpretability, $k=128$ serves as the optimal practical operating point. These results offer a transparent foundation for optimizing dimensionality and compression in biometric template engines.
