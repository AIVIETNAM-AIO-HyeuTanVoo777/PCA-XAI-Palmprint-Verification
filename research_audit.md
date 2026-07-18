# Research Audit: Multi-Level Explainable PCA in Palmprint Verification

This audit reviews the initial state of the repository, the structural specifications of the local palmprint datasets, and identifies the requirements for the explainability study.

---

## 1. Project Directory Structure
The repository is initially empty, containing only the raw dataset folders:
* `d:\FPT University\FPT_SU26\AIL303m\PCA-XAI-Palm/`
  * `ITTD Dataset/`
  * `Tongji Dataset/`

---

## 2. Dataset Specifications & Structure

### A. IIT Delhi (IITD) Touchless Palmprint Database
* **Location:** `ITTD Dataset/`
  * `Left Hand/` (Raw hands, JPG format, ~800 KB each)
  * `Right Hand/` (Raw hands, JPG format, ~800 KB each)
  * `Segmented/` (Cropped palm ROIs, BMP format, 23.8 KB each)
    * `Left/` (Files: `001_1.bmp` to `230_6.bmp`)
    * `Right/` (Files: `001_1.bmp` to `230_6.bmp`)
* **Resolution:** Cropped ROI images are $150 \times 150$ pixels, grayscale.
* **Identity Mapping:** 230 subjects, each with a left and right hand. Since left and right hand palmprints are distinct biometric signatures, they represent 460 unique classes/palms.
* **Samples per Palm:** 5 to 7 images per palm, capturing touchless variation (minor rotation, scale, and translation).

### B. Tongji Contactless Palmprint Database
* **Location:** `Tongji Dataset/`
  * `session1/` (6,000 files: `00001.bmp` to `06000.bmp`)
  * `session2/` (6,000 files: `00001.bmp` to `06000.bmp`)
* **Resolution:** $128 \times 128$ pixels, grayscale.
* **Identity Mapping:** 600 palms/classes.
* **Labeling Rule:** Every 10 consecutive images in a session correspond to one palm:
  * Images `00001.bmp` to `00010.bmp` belong to Subject 1.
  * Images `(i-1)*10 + 1` to `i*10` belong to Subject `i`.
* **Samples per Palm:** 10 images in session 1, 10 images in session 2 (total 20 samples per palm class).

---

## 3. Preprocessing & Libraries
* **Preprocessing Requirements:**
  * Unify both datasets to $128 \times 128$ pixels.
  * Normalize pixel values to $[0, 1]$.
  * Grayscale conversion (they are already grayscale BMPs, but we will ensure conversion in the loader).
  * Flattening to $16,384$-dimensional vectors for PCA fitting.
* **Required Libraries:**
  * `numpy`, `scipy` (numerical logic & EER interpolation)
  * `sklearn` (PCA fitting, metrics)
  * `pandas` (saving tabular results)
  * `cv2` (image reading/resizing)
  * `matplotlib`, `seaborn` (figure plotting)

---

## 4. Weaknesses & Improvement Opportunities
1. **Structural Inconsistency:** IITD separates left/right hands and uses subject-wise filenames (`NNN_S.bmp`). Tongji uses session-based continuous naming (`SSSSS.bmp`). 
   * *Solution:* We will write a custom loader module `dataset_loader.py` with specific adapters to unify loading APIs.
2. **Missing Directories:** The project requires separate directories for output code (`src/`), results (`results/`), figures (`figures/`), and paper files (`paper/`).
   * *Solution:* We will write `src/utils.py` to programmatically create these directories and set up directory junctions pointing from `data/IITD` and `data/Tongji` to the raw directories.
3. **Data Leakage Prevention:** Biometric evaluation requires clear split strategies:
   * *IITD:* Train on the first 3 images, test on the remaining images of each subject.
   * *Tongji:* Train on Session 1 (enrolled template), test on Session 2 (verification queries).
   This avoids random splits which leak pose/session correlations.
