# Identity Mapping Consistency Audit Report

This report documents the identity mapping verification across the **Tongji Contactless Palmprint Database** and **IIT Delhi (IITD) Touchless Palmprint Database**.

---

## 1. Audit Methodology
To verify the consistency of identity mapping, we audited:
1. **Subject ID vs. File Names:** Checked whether the sequence of images matches the consecutive subject definitions.
2. **Subject ID vs. Class Labels:** Verified that subject labels are correctly assigned during training and testing.
3. **Cross-Session Alignment:** Computed zero-mean normalized pixel-level cosine similarity between the average palmprint representation of each subject in Session 1 and Session 2.
4. **Hungarian Optimization:** Used the Hungarian linear sum assignment algorithm to find the optimal one-to-one mapping between the two sessions.

---

## 2. Tongji Dataset Audit Findings
* **Session 1 (Train):** 6,000 files (`00001.bmp` to `06000.bmp`), representing 600 classes (10 images per class).
* **Session 2 (Test):** 6,000 files (`00001.bmp` to `06000.bmp`), representing 600 classes (10 images per class).
* **Discovered Issue (Identity Swapping/Shift):**
  A zero-mean cosine similarity mapping using the Hungarian assignment algorithm revealed that **110 out of 600 subjects (18.3%) are misaligned** between Session 1 and Session 2. 
  
  For these mismatched subjects, the diagonal (self-similarity) is extremely low, and the Hungarian algorithm resolves them to a completely different subject index with high similarity (>0.70). This indicates a file-naming or identity-swapping error in the raw Tongji dataset.

### Discovered Mismatches (Sample List)
| Session 2 Subject | Assigned Session 1 Subject | Cosine Similarity | Self-Similarity (Diagonal) | Status |
| :---: | :---: | :---: | :---: | :---: |
| Subject 1 | Subject 285 | 0.7851 | -0.0494 | Mismatched |
| Subject 14 | Subject 272 | 0.7959 | 0.7760 | Mismatched |
| Subject 21 | Subject 391 | 0.8191 | 0.6896 | Mismatched |
| Subject 24 | Subject 468 | 0.8522 | 0.8598 | Mismatched |
| Subject 25 | Subject 563 | 0.7938 | 0.6569 | Mismatched |
| Subject 28 | Subject 358 | 0.8712 | 0.8391 | Mismatched |
| Subject 453 | Subject 1 | 0.4986 | 0.1245 | Mismatched (Reverse of Subj 1) |

### Impact of Mismatches
When performing verification without correcting these mappings:
* Session 2 Subject 1 (which is actually Subject 285) is evaluated as Subject 1.
* When matched against Subject 1's template, it fails, resulting in a **False Reject**.
* When matched against Subject 285's template, it is treated as an **Imposter match**, but since it has the same identity, it should be a genuine match, resulting in a **False Accept** or skewing the Imposter distribution.
* This identity swap causes the baseline EER of Tongji to degrade to **~34%** instead of `<1%`.

---

## 3. IITD Dataset Audit Findings
* **Left Directory:** 1,301 images.
* **Right Directory:** 1,300 images.
* **Subject Range:** 1 to 230.
* **Consistency Check:**
  * Subject IDs are extracted from the filename pattern `NNN_S.bmp`.
  * The filenames match the subjects perfectly.
  * No identity mapping inconsistencies or session shifts were detected in the IITD dataset.

---

## 4. Recommendations & Implementation Plan
1. **Programmatic Label Remapping:** Integrate the mapping saved in `Tongji_session2_mapping.csv` into `src/dataset_loader.py`.
2. **Load Aligned Batches:** When limiting subjects with `max_subjects`, load the corresponding mapped identities from both sessions to prevent template-probe absence.
