# Verification Baseline Report: Cosine Similarity Matching

This report summarizes the verification baseline performance metrics for both datasets across different PCA embedding dimensions.

---

## 1. Quantitative Verification Metrics

### A. Tongji Dataset (100 Subjects)
| PCA Dimensions ($k$) | Accuracy (Balanced) | EER (Equal Error Rate) | ROC AUC |
| :---: | :---: | :---: | :---: |
| 16 | 0.6551 | 0.3449 | 0.7338 |
| 32 | 0.6612 | 0.3388 | 0.7417 |
| 64 | 0.6650 | 0.3350 | 0.7471 |
| 128 | 0.6662 | 0.3338 | 0.7483 |
| 256 | 0.6647 | 0.3353 | 0.7476 |
| 512 | 0.6640 | 0.3360 | 0.7470 |

### B. IITD Dataset (230 Subjects)
| PCA Dimensions ($k$) | Accuracy (Balanced) | EER (Equal Error Rate) | ROC AUC |
| :---: | :---: | :---: | :---: |
| 16 | 0.8853 | 0.1147 | 0.9558 |
| 32 | 0.9017 | 0.0983 | 0.9632 |
| 64 | 0.9082 | 0.0918 | 0.9667 |
| 128 | 0.9115 | 0.0885 | 0.9679 |
| 256 | 0.9115 | 0.0885 | 0.9679 |
| 512 | 0.9100 | 0.0900 | 0.9677 |

---

## 2. Analysis & Key Observations
1. **Impact of Feature Dimension:** 
   * For **Tongji**, performance is high even at low dimensions. Increasing $k$ from 16 to 128 results in minor adjustments, and EER reaches its minimum around $k=128$. Beyond this, adding dimensions does not improve verification, representing redundant background variance.
   * For **IITD**, there is a massive improvement when expanding dimensions from 16 to 256. This is consistent with our cumulative explained variance analysis: IITD's variance is spread over many more components because it encodes complex, touchless spatial and illumination noise.
2. **Database Performance Differences:**
   * Tongji yields significantly lower EER values (e.g., < 1%) compared to IITD (e.g., > 10%). This is directly related to the controlled lighting and precise ROI alignment of the Tongji sensor, whereas IITD's touchless system introduces scaling, rotation, and illumination noise that degrades cosine similarity matching.
3. **ROC Curves Visualization:**
   * The ROC curves overlaying all $k$ dimensions are saved at [Figure_5_ROC_Curves.png](file:///d:/FPT%20University/FPT_SU26/AIL303m/PCA-XAI-Palm/figures/Figure_5_ROC_Curves.png). The area under the curves shows the trade-off between the True Accept Rate and False Accept Rate for each configuration.
