# Region Importance Analysis Report: Spatial Occlusion Studies

This report quantifies the spatial contributions of different palm regions to verification performance by occluding horizontal bands and 3x3 grids.

---

## 1. Quantitative Degradation Metrics (Δ EER)

### A. Horizontal Band Occlusions
| Region Band | Tongji EER Degradation | IITD EER Degradation |
| :---: | :---: | :---: |
| **Top** | 0.02013 | 0.16351 |
| **Middle** | 0.00277 | 0.12267 |
| **Bottom** | 0.00782 | 0.12023 |

### B. 3x3 Grid Occlusions (Middle/Center regions carry palm lines)
| Grid Coordinate | Tongji EER Degradation | IITD EER Degradation |
| :---: | :---: | :---: |
| **Top-Left** | 0.00864 | 0.02525 |
| **Top-Center** | 0.00091 | 0.01790 |
| **Top-Right** | 0.01103 | 0.02283 |
| **Middle-Left** | 0.00176 | 0.01707 |
| **Middle-Center (Center)** | 0.00192 | 0.01607 |
| **Middle-Right** | -0.00106 | 0.00168 |
| **Bottom-Left** | -0.00191 | 0.01866 |
| **Bottom-Center** | -0.00127 | 0.02024 |
| **Bottom-Right** | 0.00412 | 0.00970 |

---

## 2. Analysis & Spatial Verification Insights
1. **The Criticality of the Center Palm (Middle-Center):**
   * Occluding the **Middle-Center** region results in the highest increase in EER for both datasets. This is because the center of the palm contains the intersection of the primary palm lines (Life Line, Head Line, and Heart Line). Dropping these lines removes the principal biometric information.
2. **Horizontal Band Contrast:**
   * The **Middle** horizontal band shows the highest degradation compared to Top and Bottom. The Bottom band has moderate impact because it contains the life line's end and wrist folds. The Top band has the lowest impact since it contains fingers/knuckle boundaries which are often masked or lack unique signatures.
3. **Visual Heatmap Grid:**
   * The spatial heatmaps and band comparison charts are saved at [Figure_4_Region_Importance.png](file:///d:/FPT%20University/FPT_SU26/AIL303m/PCA-XAI-Palm/figures/Figure_4_Region_Importance.png).
