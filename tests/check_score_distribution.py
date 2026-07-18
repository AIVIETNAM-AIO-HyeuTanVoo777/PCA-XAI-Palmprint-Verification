import os
import glob
import cv2
import numpy as np

def check_dist():
    sess1_dir = "Tongji Dataset/session1"
    sess2_dir = "Tongji Dataset/session2"
    
    sess1_files = sorted(glob.glob(os.path.join(sess1_dir, "*.bmp")))
    sess2_files = sorted(glob.glob(os.path.join(sess2_dir, "*.bmp")))
    
    # We will pick a few subjects that we believe are aligned
    # E.g. Subject 3 (index 2), Subject 5 (index 4), Subject 7 (index 6)
    aligned_indices = [2, 4, 6, 8, 12, 14, 16]
    
    for idx in aligned_indices:
        # Load Session 2 images for idx
        block_s2 = []
        for f in sess2_files[idx*10 : (idx+1)*10]:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            block_s2.append(cv2.resize(img, (64, 64)).astype(np.float32) / 255.0)
        s2_mean = np.mean(block_s2, axis=0).flatten()
        s2_zm = s2_mean - np.mean(s2_mean)
        s2_norm = s2_zm / (np.linalg.norm(s2_zm) + 1e-9)
        
        # Calculate similarity with Session 1 idx
        block_s1 = []
        for f in sess1_files[idx*10 : (idx+1)*10]:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            block_s1.append(cv2.resize(img, (64, 64)).astype(np.float32) / 255.0)
        s1_mean = np.mean(block_s1, axis=0).flatten()
        s1_zm = s1_mean - np.mean(s1_mean)
        s1_norm = s1_zm / (np.linalg.norm(s1_zm) + 1e-9)
        
        self_sim = s2_norm @ s1_norm
        
        # Calculate similarity with all other Session 1 subjects
        other_sims = []
        for other_idx in range(600):
            if other_idx == idx:
                continue
            block_other = []
            for f in sess1_files[other_idx*10 : (other_idx+1)*10]:
                img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    block_other.append(cv2.resize(img, (64, 64)).astype(np.float32) / 255.0)
            other_mean = np.mean(block_other, axis=0).flatten()
            other_zm = other_mean - np.mean(other_mean)
            other_norm = other_zm / (np.linalg.norm(other_zm) + 1e-9)
            
            other_sims.append(s2_norm @ other_norm)
            
        print(f"Subject {idx+1}:")
        print(f"  Self-similarity (genuine): {self_sim:.4f}")
        print(f"  Imposter similarities - Mean: {np.mean(other_sims):.4f}, Max: {np.max(other_sims):.4f}, Std: {np.std(other_sims):.4f}")

if __name__ == "__main__":
    check_dist()
