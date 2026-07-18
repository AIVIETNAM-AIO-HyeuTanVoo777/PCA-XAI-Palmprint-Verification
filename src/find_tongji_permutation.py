import os
import glob
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

def find_permutation():
    sess1_dir = "Tongji Dataset/session1"
    sess2_dir = "Tongji Dataset/session2"
    
    sess1_files = sorted(glob.glob(os.path.join(sess1_dir, "*.bmp")))
    sess2_files = sorted(glob.glob(os.path.join(sess2_dir, "*.bmp")))
    
    n_subjects = len(sess1_files) // 10
    
    s1_images = []
    s2_images = []
    
    print("Loading all images...")
    for i in range(n_subjects):
        block_s1 = []
        for f in sess1_files[i*10 : (i+1)*10]:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                block_s1.append(cv2.resize(img, (64, 64)).astype(np.float32) / 255.0)
        s1_images.append(np.mean(block_s1, axis=0).flatten() if block_s1 else np.zeros(4096))
        
        block_s2 = []
        for f in sess2_files[i*10 : (i+1)*10]:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                block_s2.append(cv2.resize(img, (64, 64)).astype(np.float32) / 255.0)
        s2_images.append(np.mean(block_s2, axis=0).flatten() if block_s2 else np.zeros(4096))
        
    s1_images = np.array(s1_images)
    s2_images = np.array(s2_images)
    
    # Zero-mean normalization
    s1_zm = s1_images - np.mean(s1_images, axis=1, keepdims=True)
    s2_zm = s2_images - np.mean(s2_images, axis=1, keepdims=True)
    
    s1_norms = s1_zm / (np.linalg.norm(s1_zm, axis=1, keepdims=True) + 1e-9)
    s2_norms = s2_zm / (np.linalg.norm(s2_zm, axis=1, keepdims=True) + 1e-9)
    
    # Calculate similarity matrix (Sess1 x Sess2)
    sims = s1_norms @ s2_norms.T
    
    # Linear sum assignment (Hungarian algorithm) to find one-to-one mapping that maximizes total similarity
    # Cost matrix is negative similarity
    row_ind, col_ind = linear_sum_assignment(-sims)
    
    # row_ind corresponds to Session 1 index, col_ind corresponds to Session 2 index
    # Let's map Session 2 index to Session 1 index
    s2_to_s1_map = {}
    mismatches = 0
    correct_matches = 0
    
    for r, c in zip(row_ind, col_ind):
        # r is s1 subject index (0-based)
        # c is s2 subject index (0-based)
        sim = sims[r, c]
        s2_to_s1_map[c] = (r, sim)
        
        if r == c:
            correct_matches += 1
        else:
            mismatches += 1
            if mismatches <= 30:
                print(f"Assigning Session 2 Subj {c+1} -> Session 1 Subj {r+1} (sim: {sim:.4f})")
                
    print(f"\nLinear Assignment Summary:")
    print(f"Total subjects: {n_subjects}")
    print(f"Matched on diagonal: {correct_matches}")
    print(f"Mismatches (swaps): {mismatches}")
    
    # Let's save the mapping to a file so that we can load it during verification
    # We can write a dictionary mapping from session 2 subject idx to correct label
    mapping_data = []
    for c in range(n_subjects):
        r, sim = s2_to_s1_map[c]
        mapping_data.append(f"{c},{r},{sim}")
        
    with open("Tongji_session2_mapping.csv", "w") as f:
        f.write("session2_index,session1_index,similarity\n")
        f.write("\n".join(mapping_data))
    print("Saved mapping to Tongji_session2_mapping.csv")

if __name__ == "__main__":
    find_permutation()
