# Verification Pipeline & Metric Validation Report

This report audits the verification pipeline and provides the formal mathematical derivations for the metrics used to evaluate the palmprint verification performance: **Accuracy, ROC, AUC, EER, FAR, and FRR**.

---

## 1. Mathematical Derivations of Metrics

### A. Similarity Metric: Cosine Similarity
Given a probe feature vector $x \in \mathbb{R}^k$ and a template feature vector $t \in \mathbb{R}^k$, their match score $s(x, t)$ is defined by the Cosine Similarity:
$$s(x, t) = \frac{x \cdot t}{\|x\|_2 \|t\|_2} = \frac{\sum_{i=1}^k x_i t_i}{\sqrt{\sum_{i=1}^k x_i^2} \sqrt{\sum_{i=1}^k t_i^2}}$$
Where:
* $s(x, t) \in [-1, 1]$.
* Since palmprint feature coordinates are typically positive (or normalized), the similarity score is usually in $[0, 1]$. Higher similarity indicates a closer match.

### B. False Accept Rate (FAR)
FAR is the probability that the system incorrectly accepts an impostor probe (matching it to a template of a different subject). For a similarity threshold $\theta$:
$$FAR(\theta) = P(s(x_{imp}, t) \ge \theta) = \frac{|\{s \in S_{imp} : s \ge \theta\}|}{|S_{imp}|}$$
Where:
* $S_{imp}$ is the set of all impostor matching scores.
* $|S_{imp}|$ is the total number of impostor comparisons.

### C. False Reject Rate (FRR)
FRR is the probability that the system incorrectly rejects a genuine probe (failing to match it to its own template). For a similarity threshold $\theta$:
$$FRR(\theta) = P(s(x_{gen}, t) < \theta) = \frac{|\{s \in S_{gen} : s < \theta\}|}{|S_{gen}|}$$
Where:
* $S_{gen}$ is the set of all genuine matching scores.
* $|S_{gen}|$ is the total number of genuine comparisons.
* The True Accept Rate (TAR) is defined as $TAR(\theta) = 1 - FRR(\theta)$.

### D. Equal Error Rate (EER)
The Equal Error Rate is the operational point where the False Accept Rate equals the False Reject Rate:
$$EER = FAR(\theta_{EER}) = FRR(\theta_{EER})$$
Subject to:
$$\theta_{EER} = \arg\min_{\theta} |FAR(\theta) - FRR(\theta)|$$
In practice, because similarity scores are discrete, $FAR(\theta)$ and $FRR(\theta)$ curves might not intersect at an exact sample point. We interpolate or take the average at the point of closest approach:
$$EER = \frac{FAR(\theta^*) + FRR(\theta^*)}{2}, \quad \text{where } \theta^* = \arg\min_{\theta} |FAR(\theta) - FRR(\theta)|$$

### E. Balanced Accuracy
Balanced Accuracy measures the overall classification correctness, balancing genuine and imposter classes. At the EER operational threshold $\theta_{EER}$:
$$\text{Accuracy} = 1 - EER$$
This represents the probability of correct verification under equal class priors.

### F. Receiver Operating Characteristic (ROC) & Area Under Curve (AUC)
* **ROC Curve:** A parametric plot of the True Accept Rate ($TAR(\theta) = 1 - FRR(\theta)$) on the y-axis against the False Accept Rate ($FAR(\theta)$) on the x-axis, for all possible thresholds $\theta \in [-1, 1]$.
* **AUC:** The area under the ROC curve, computed via integration:
$$AUC = \int_{0}^{1} TAR(FAR^{-1}(u)) \, du$$
An AUC of $1.0$ represents perfect verification; $0.5$ represents random guessing.

---

## 2. Code Pipeline Audit
We inspected the verification pipeline implemented in `src/verification.py`:

1. **Feature Extraction & Projection:**
   * Images are flattened and projected using scikit-learn's `PCA.transform()`: $Z = (X - \mu) W_k$, where $W_k$ is the matrix of the top $k$ eigenvectors. (Correct)
2. **Template Representation:**
   * Class template is computed as the mean projection: $t_c = \frac{1}{N_{train}} \sum_{i \in c} z_i$. (Correct)
3. **Similarity Score Generation:**
   * Probes and templates are normalized: $\hat{z} = \frac{z}{\|z\|}$, $\hat{t} = \frac{t}{\|t\|}$.
   * Similarity matrix is computed via matrix multiplication: $S = \hat{Z}_{test} \hat{T}^T$. (Correct and computationally efficient)
4. **EER Computation Logic:**
   * The code calls `roc_curve(y_true, y_scores, pos_label=1)`.
   * It extracts `fpr` (which is $FAR$) and `fnr = 1 - tpr` (which is $FRR$).
   * It finds the index where $|fpr - fnr|$ is minimized, and averages them: `eer = (fpr[idx] + fnr[idx]) / 2.0`. (Correct and numerically stable)

---

## 3. Findings & Recommendations
The pipeline is mathematically sound and follows biometrics standards.
* **Minor suggestion:** To prevent division by zero during normalization, the code uses `np.clip(..., 1e-9, None)`. This is a robust practice for numeric stability.
* **Seed Determinism:** The PCA object is instantiated with `random_state=42`, which guarantees deterministic eigen-decomposition across runs.
