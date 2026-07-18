import os
import glob
import cv2
import numpy as np

def test_match():
    sess1_dir = "Tongji Dataset/session1"
    sess2_dir = "Tongji Dataset/session2"
    
    sess1_files = sorted(glob.glob(os.path.join(sess1_dir, "*.bmp")))
    sess2_files = sorted(glob.glob(os.path.join(sess2_dir, "*.bmp")))
    
    # Load Subject 1 of Session 2 (files 0-9)
    block_s2 = []
    for f in sess2_files[0:10]:
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        block_s2.append(cv2.resize(img, (128, 128)).astype(np.float32) / 255.0)
    s2_mean = np.mean(block_s2, axis=0).flatten()
    s2_zm = s2_mean - np.mean(s2_mean)
    s2_norm = s2_zm / (np.linalg.norm(s2_zm) + 1e-9)
    
    # Compute similarity with all 600 subjects in Session 1
    sims = []
    for i in range(600):
        block_s1 = []
        for f in sess1_files[i*10 : (i+1)*10]:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                block_s1.append(cv2.resize(img, (128, 128)).astype(np.float32) / 255.0)
        s1_mean = np.mean(block_s1, axis=0).flatten()
        s1_zm = s1_mean - np.mean(s1_mean)
        s1_norm = s1_zm / (np.linalg.norm(s1_zm) + 1e-9)
        
        sim = s2_norm @ s1_norm
        sims.append((i, sim))
        
    # Sort by similarity descending
    sims.sort(key=lambda x: x[1], reverse=True)
    
    print("Top 10 matches for Session 2 Subject 1:")
    for idx, sim in sims[:10]:
        print(f"  Session 1 Subject {idx+1} (files {idx*10+1:05d}-{idx*10+10:05d}.bmp): similarity = {sim:.4f}")
        
if __name__ == "__main__":
    test_match()
