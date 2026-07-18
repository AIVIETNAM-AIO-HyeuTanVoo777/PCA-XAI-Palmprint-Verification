# Session Split & Protocol Verification Report

This report documents the verification of the session splits, train/test divisions, and template/probe protocols across the **Tongji Contactless Palmprint Database** and **IIT Delhi (IITD) Touchless Palmprint Database**.

---

## 1. Tongji Dataset Protocol Verification
The Tongji dataset was collected in two distinct sessions with an average interval of 29 days between sessions. This temporal separation is critical for evaluating the robustness of biometric features against time-lapse variations (e.g., healing of minor cuts, changes in skin hydration, and minor position shifts).

* **Intended Protocol:**
  * **Session 1:** Used for enrollment (template generation) and training.
  * **Session 2:** Used for verification (probes).
* **Code Implementation Verification:**
  * In `src/dataset_loader.py`:
    * Train set ($X_{train}, y_{train}$) is loaded exclusively from `Tongji Dataset/session1`.
    * Test set ($X_{test}, y_{test}$) is loaded exclusively from `Tongji Dataset/session2`.
  * **Template Generation:**
    * During evaluation in `src/verification.py`, a single template vector is computed for each class by taking the mean of its PCA projections in the training set (Session 1).
  * **Probe Matching:**
    * Every test sample from Session 2 (probe) is matched against all class templates.
  * **Split Count:**
    * For $N$ subjects, we load $N \times 10$ training images (Session 1) and $N \times 10$ test images (Session 2).
* **Protocol Match Status:** **PASSED** (strictly follows the temporal session split).

---

## 2. IITD Dataset Protocol Verification
The IITD dataset does not have explicit temporal session subdirectories. Instead, it consists of a single set of images per palm.

* **Intended Protocol:**
  * To simulate enrollment, the first $M$ sorted images of each palm are used as the gallery (training/templates).
  * The remaining images are used as probes (testing).
  * Standard partition: $M = 3$ images for training, and the remaining (typically 2 to 4) images for testing.
* **Code Implementation Verification:**
  * In `src/dataset_loader.py`:
    * For each palm class (e.g. `1_L`), the files are sorted by sample ID (extracted from `NNN_S.bmp`).
    * The first 3 sorted files are assigned to `X_train`.
    * The remaining files (indices 3 onwards) are assigned to `X_test`.
  * **Template Generation:**
    * During evaluation in `src/verification.py`, the template is the mean PCA projection of the 3 training samples.
  * **Probe Matching:**
    * All test samples (probes) are matched against the 460 templates using cosine similarity.
* **Protocol Match Status:** **PASSED** (strictly follows the standard first-3 train, remaining-test partitioning protocol).

---

## 3. Genuine & Imposter Score Count Verification
For verification experiments, the number of genuine and imposter comparisons must be verified to ensure correct statistical power.

### A. Tongji Dataset ($N = 100$ subjects, 10 probes per subject)
* **Total Probes (Test samples):** $100 \times 10 = 1,000$.
* **Templates (Classes):** 100.
* **Genuine Comparisons:** Each of the 1,000 probes is matched against its own class template $\rightarrow$ **1,000 genuine scores**.
* **Imposter Comparisons:** Each of the 1,000 probes is matched against the 99 non-matching templates $\rightarrow$ $1,000 \times 99 =$ **99,000 imposter scores**.

### B. IITD Dataset ($N = 460$ palm classes)
* **Total Probes (Test samples):** The total number of test files is $N_{test} = \sum_{c} (S_c - 3)$, where $S_c$ is the number of samples for class $c$ (between 5 and 7).
  * For 460 classes, with average 5.6 samples per class:
  * Total samples: 2,601 (1,301 Left + 1,300 Right).
  * Train samples: $460 \times 3 = 1,380$.
  * Test samples (probes): $2,601 - 1,380 = 1,221$.
* **Genuine Comparisons:** Each probe matches its own template $\rightarrow$ **1,221 genuine scores**.
* **Imposter Comparisons:** Each probe matches the 459 non-matching templates $\rightarrow$ $1,221 \times 459 =$ **560,439 imposter scores**.

This matches the standard closed-set verification evaluation protocol.
