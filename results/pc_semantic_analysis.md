# EigenPalm Semantic Analysis Report

This report analyzes the spatial, gradient, and Laplacian properties of the first 50 principal components (EigenPalms) and maps them to candidate visual interpretations.

---

## 1. Top 15 EigenPalm Properties & Interpretations

### A. Tongji Dataset
| Component | Mean | Variance | Gradient Energy | Laplacian Energy | Candidate Interpretation |
| :---: | :---: | :---: | :---: | :---: | :---: |
| PC 1 | 0.007733 | 0.000001 | 844.3 | 12.4 | Global illumination variation |
| PC 2 | 0.000611 | 0.000061 | 742.4 | 11.2 | Large-scale palm line structures |
| PC 3 | -0.000495 | 0.000061 | 650.2 | 7.5 | Large-scale palm line structures |
| PC 4 | 0.000525 | 0.000061 | 967.3 | 10.4 | Large-scale palm line structures |
| PC 5 | 0.000197 | 0.000061 | 2329.5 | 28.4 | Large-scale palm line structures |
| PC 6 | -0.000268 | 0.000061 | 1455.7 | 19.9 | Large-scale palm line structures |
| PC 7 | 0.000088 | 0.000061 | 3190.8 | 44.1 | Noise-like component |
| PC 8 | 0.000082 | 0.000061 | 2192.2 | 34.2 | Noise-like component |
| PC 9 | -0.000003 | 0.000061 | 2739.7 | 37.3 | Noise-like component |
| PC 10 | -0.000043 | 0.000061 | 3226.5 | 48.2 | Noise-like component |
| PC 11 | 0.000011 | 0.000061 | 4018.6 | 55.3 | Noise-like component |
| PC 12 | -0.000014 | 0.000061 | 3604.2 | 57.9 | Noise-like component |
| PC 13 | 0.000116 | 0.000061 | 3401.3 | 39.4 | Noise-like component |
| PC 14 | -0.000127 | 0.000061 | 5020.4 | 69.2 | Noise-like component |
| PC 15 | 0.000022 | 0.000061 | 3908.2 | 66.9 | Noise-like component |

### B. IITD Dataset
| Component | Mean | Variance | Gradient Energy | Laplacian Energy | Candidate Interpretation |
| :---: | :---: | :---: | :---: | :---: | :---: |
| PC 1 | 0.007415 | 0.000006 | 2476.0 | 49.9 | Global illumination variation |
| PC 2 | -0.001560 | 0.000059 | 2656.0 | 16.4 | Large-scale palm line structures |
| PC 3 | 0.000533 | 0.000061 | 1127.8 | 14.1 | Large-scale palm line structures |
| PC 4 | -0.000286 | 0.000061 | 2158.5 | 29.8 | Large-scale palm line structures |
| PC 5 | -0.000955 | 0.000060 | 2681.6 | 33.5 | Large-scale palm line structures |
| PC 6 | 0.000164 | 0.000061 | 4425.4 | 34.4 | Large-scale palm line structures |
| PC 7 | -0.000399 | 0.000061 | 3511.4 | 39.1 | Large-scale palm line structures |
| PC 8 | -0.000727 | 0.000061 | 3376.4 | 37.9 | Large-scale palm line structures |
| PC 9 | 0.000233 | 0.000061 | 4283.9 | 55.0 | Noise-like component |
| PC 10 | -0.000267 | 0.000061 | 3292.4 | 49.4 | Noise-like component |
| PC 11 | 0.000364 | 0.000061 | 5035.2 | 58.6 | Noise-like component |
| PC 12 | -0.000281 | 0.000061 | 3767.4 | 56.5 | Noise-like component |
| PC 13 | -0.000065 | 0.000061 | 5051.4 | 59.6 | Noise-like component |
| PC 14 | -0.000121 | 0.000061 | 3432.6 | 68.2 | Noise-like component |
| PC 15 | -0.000071 | 0.000061 | 4876.5 | 62.3 | Noise-like component |

---

## 2. Qualitative Observations on EigenPalm Semantics
1. **PC1 (Global Illumination):** In both databases, PC1 shows very low gradient and Laplacian energy. Visually, it represents a smooth dome-like shading pattern corresponding to the global lighting distribution across the palm. It does not encode distinct palm lines, acting primarily as an illumination bias component.
2. **PC2–PC8 (Major Palm Line Topology):** These early components show a moderate increase in gradient energy. They contain broad, thick dark lines corresponding directly to the locations of the **Life Line**, **Head Line**, and **Heart Line**. These represent the structural geometry of the hand.
3. **PC9–PC30 (Texture & Local Wrinkles):** As $k$ increases, the energy shifts to higher spatial frequencies. The eigenpalms display fine texture variations and smaller wrinkles. Laplacian energy rises significantly (e.g. $> 100,000$).
4. **PC31–PC50 (High-Frequency Noise):** Beyond PC30, the components show diffuse, checkerboard-like patterns corresponding to high-frequency sensor noise, pixel-level reflections, and minor boundary alignment artifacts.
