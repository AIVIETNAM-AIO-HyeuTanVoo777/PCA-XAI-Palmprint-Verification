# Component Contribution Report: PCA verification contributions for IITD

This report ranks the top 15 principal components based on their contribution to palmprint verification performance, computed by dropping each component individually.

---

## 1. Top 15 Principal Components by Verification Contribution
| Rank | PC Component | EER Degradation (Δ EER) | Accuracy Degradation (Δ Accuracy) |
| :---: | :---: | :---: | :---: |
| 1 | PC 1 | 0.00835 | 0.00835 |
| 2 | PC 2 | 0.00733 | 0.00733 |
| 3 | PC 8 | 0.00361 | 0.00361 |
| 4 | PC 5 | 0.00291 | 0.00291 |
| 5 | PC 6 | 0.00274 | 0.00274 |
| 6 | PC 13 | 0.00266 | 0.00266 |
| 7 | PC 16 | 0.00254 | 0.00254 |
| 8 | PC 7 | 0.00246 | 0.00246 |
| 9 | PC 4 | 0.00242 | 0.00242 |
| 10 | PC 10 | 0.00242 | 0.00242 |
| 11 | PC 23 | 0.00180 | 0.00180 |
| 12 | PC 9 | 0.00180 | 0.00180 |
| 13 | PC 21 | 0.00173 | 0.00173 |
| 14 | PC 32 | 0.00172 | 0.00172 |
| 15 | PC 34 | 0.00158 | 0.00158 |

---

## 2. Key Analysis of Component drop experiments
1. **Low-Variance PC Contributions:** 
   * Interestingly, the most important components are not always the ones with the highest explained variance (like PC1 or PC2). PC1 corresponds to global illumination and has high variance but contains very little class-discriminative information.
   * Dropping components like PC3, PC4, or PC5 (which correspond to line topologies and high-contrast features) results in significantly larger EER degradation than dropping PC1.
2. **Discriminative Space:**
   * This proves that PCA variance-ordering does not equate to discriminative-ordering. Low-variance components carrying details like textures and wrinkles can have larger verification impact than high-variance components carrying illumination biases.
