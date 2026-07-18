import os
import glob
import cv2
import numpy as np

def audit_tongji():
    print("=== Auditing Tongji ===")
    sess1_dir = "Tongji Dataset/session1"
    sess2_dir = "Tongji Dataset/session2"
    
    if not os.path.exists(sess1_dir) or not os.path.exists(sess2_dir):
        print("Tongji directories not found.")
        return
        
    sess1_files = sorted(glob.glob(os.path.join(sess1_dir, "*.bmp")))
    sess2_files = sorted(glob.glob(os.path.join(sess2_dir, "*.bmp")))
    
    n_blocks = min(len(sess1_files) // 10, 600)
    print(f"Checking first {n_blocks} subjects for alignment...")
    
    s1_images = []
    s2_images = []
    
    for i in range(n_blocks):
        block_s1 = []
        for f in sess1_files[i*10 : (i+1)*10]:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_res = cv2.resize(img, (32, 32)).astype(np.float32) / 255.0
                block_s1.append(img_res)
        s1_images.append(np.mean(block_s1, axis=0).flatten() if block_s1 else np.zeros(1024))
        
        block_s2 = []
        for f in sess2_files[i*10 : (i+1)*10]:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_res = cv2.resize(img, (32, 32)).astype(np.float32) / 255.0
                block_s2.append(img_res)
        s2_images.append(np.mean(block_s2, axis=0).flatten() if block_s2 else np.zeros(1024))
        
    s1_images = np.array(s1_images)
    s2_images = np.array(s2_images)
    
    # Subtract mean of each image to look at fine features/textures instead of global average intensity
    s1_zm = s1_images - np.mean(s1_images, axis=1, keepdims=True)
    s2_zm = s2_images - np.mean(s2_images, axis=1, keepdims=True)
    
    s1_norms = s1_zm / (np.linalg.norm(s1_zm, axis=1, keepdims=True) + 1e-9)
    s2_norms = s2_zm / (np.linalg.norm(s2_zm, axis=1, keepdims=True) + 1e-9)
    
    sims = s1_norms @ s2_norms.T
    
    mismatches = []
    for i in range(n_blocks):
        self_sim = sims[i, i]
        best_match_idx = np.argmax(sims[i, :])
        best_sim = sims[i, best_match_idx]
        
        # Check if the correct match is NOT the diagonal
        if best_match_idx != i or self_sim < 0.3:
            mismatches.append((i, self_sim, best_match_idx, best_sim))
            
    print(f"Found {len(mismatches)} subjects with potential alignment issues:")
    for idx, self_sim, best_idx, best_sim in mismatches[:30]:
        print(f"  Session 2 Subj {idx+1} (files {idx*10+1:05d}-{idx*10+10:05d}.bmp) matches Session 1 Subj {best_idx+1} ({best_sim:.4f}) with self similarity {self_sim:.4f}")
        
    return mismatches, sims

if __name__ == "__main__":
    audit_tongji()
