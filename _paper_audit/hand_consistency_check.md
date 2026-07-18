# Hand Consistency & Separation Audit Report

This report documents the verification of hand consistency and left/right palm separation across the **IIT Delhi (IITD) Touchless Palmprint Database** and the **Tongji Contactless Palmprint Database**.

---

## 1. Context & Importance
Biometric studies have shown that left and right palmprint patterns for the same subject are genetically similar but physically distinct (analogous to the iris patterns or fingerprints of identical twins). 
Treating left and right palms from the same subject as the same biometric identity:
1. **Lacks Biological Validity:** They possess unique principal lines, wrinkles, and textures.
2. **Artificially Inflates Performance:** Matching a left palm against a right palm of the same subject will produce low similarity. If they are labeled as the same identity, it creates a massive number of false rejects.
3. **Leads to Code Defects:** It violates the core assumption of biometric verification systems where each class corresponds to a unique physical sensor signature.

---

## 2. IITD Dataset Hand Consistency Audit
* **Directory Structure:**
  * `IITD Dataset/Segmented/Left` contains 1,301 images from 230 subjects.
  * `IITD Dataset/Segmented/Right` contains 1,300 images from 230 subjects.
* **Audit of Hand Classification:**
  * In the data loading module `src/dataset_loader.py`, we verify that the loader constructs a class label by combining the Subject ID (e.g. `1`) and the hand side (`L` or `R`).
  * Example keys: `1_L` and `1_R` are mapped to different class indices (e.g., class 0 and class 1).
  * This results in a total of **460 distinct palm classes** for the 230 subjects.
* **Verification of Separation:**
  * We verified that no image from the `Left` directory is ever assigned to a `Right` palm class or vice versa.
  * The class indexing is deterministic and based on sorted keys: `(Subject_ID, Hand_Side)`.
  * Thus, left and right palms are strictly treated as **separate identities**, preventing any cross-hand identity leaks.

---

## 3. Tongji Dataset Hand Consistency Audit
* **Directory Structure:**
  * `Tongji Dataset/session1` and `session2` contain images labeled `00001.bmp` to `06000.bmp`.
  * The Tongji database was collected from 300 subjects. For each subject, both the left and right palms were captured (representing 600 unique palms).
* **Audit of Hand Classification:**
  * The dataset is structured such that each of the 600 unique palms is treated as a separate subject directory/block.
  * Specifically, files are grouped in blocks of 10. Images 1–10 correspond to Subject 1 (e.g. Subject 1's Left palm), images 11–20 correspond to Subject 2 (e.g. Subject 1's Right palm), and so on.
  * Our loader groups images in blocks of 10 consecutive files, resulting in **600 distinct palm classes**.
* **Verification of Separation:**
  * There are no files from different palms mixed in the same 10-image block, except for the cross-session index mismatches discussed in the [Identity Mapping Audit](file:///d:/FPT%20University/FPT_SU26/AIL303m/PCA-XAI-Palm/reports/identity_mapping_check.md).
  * With programmatic label remapping, the left and right palms remain strictly separated and correctly mapped to their unique Session 1 templates.

---

## 4. Conclusion
Both datasets successfully maintain left/right palm separation:
* **IITD:** Programmatic separation of `L` and `R` files yields 460 unique classes.
* **Tongji:** Consecutive block loading of 10 images yields 600 unique classes.
No identity leaks or invalid cross-hand matchings occur in the verification pipeline.
