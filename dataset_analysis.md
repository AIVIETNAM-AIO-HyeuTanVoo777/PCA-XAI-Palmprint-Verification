# Dataset Analysis: Tongji vs. IITD Touchless Palmprints

This report provides a comparative analysis of the **Tongji Contactless Palmprint Dataset** and the **IIT Delhi (IITD) Touchless Palmprint Dataset** used in this multi-level explainability study.

---

## 1. Summary Statistics
The dataset summary statistics are logged dynamically during the data loading stage and saved to `results/dataset_summary.csv`. The summary values are as follows:

| Dataset | Subjects | Palms (Classes) | Total Images | Train Samples/Class | Test Samples/Class | Original ROI Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tongji** | 100 | 100 | 2,000 | 10 (Session 1) | 10 (Session 2) | $128 \times 128$ |
| **IITD** | 230 | 460 | 2,601 | 3 (Sorted 1-3) | 2–4 (Sorted 4+) | $150 \times 150$ |

*Note: For Tongji, we evaluate on a standard subset of 100 subjects to optimize compute efficiency and ensure SHAP analysis completes without memory overflow, while for IITD we evaluate on all 230 available subjects (460 unique palm classes).*

---

## 2. Acquisition Conditions & Image Quality

### A. Tongji Dataset
* **Acquisition Device:** Contactless palmprint capture system with a circular fluorescent light source. The system is designed to allow free hand movement.
* **Image Quality:** High contrast and sharp features. The principal lines (life, head, heart) and fine wrinkles are clearly defined against the skin texture.
* **Illumination Variation:** Very low, due to the controlled circular light source. The intensity distribution across the palm is smooth, with the center of the palm being slightly brighter because it is closer to the light source.
* **ROI Consistency:** Excellent. The pre-cropped ROI images are highly centered and scale-normalized, minimizing global geometric variations.

### B. IITD Dataset
* **Acquisition Device:** Touchless palmprint imaging system with a circular fluorescent illuminator. The database contains hand images of volunteers in indoor environment.
* **Image Quality:** Good quality, but slightly lower contrast compared to Tongji. The cropped segmented ROIs are extracted from the raw hands using a standard keypoint-based alignment algorithm.
* **Illumination Variation:** Moderate. Some images show highlight reflections or minor shadows at the boundaries due to indoor lighting variations and differences in skin reflection properties.
* **ROI Consistency:** Good, but exhibits minor translation and rotation variations. Because the acquisition is touchless and doesn't restrict hand posture, minor out-of-plane rotation is present in several samples.

---

## 3. Sample Visualizations
A grid of preprocessed samples ($128 \times 128$ grayscale) from both datasets is saved at:
* [Figure_2_DatasetSamples.png](file:///d:/FPT%20University/FPT_SU26/AIL303m/PCA-XAI-Palm/figures/Figure_2_DatasetSamples.png)

This figure shows:
* **Row 1 (Tongji):** Clear principal lines with minimal background noise, high center-alignment, and consistent lighting.
* **Row 2 (IITD):** Detailed texture but with slight translation offsets and slight variance in illumination contrast across the palm center.
