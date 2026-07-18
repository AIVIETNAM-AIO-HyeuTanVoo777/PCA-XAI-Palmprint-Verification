import os
import glob
import cv2
import numpy as np

def test_raw_sim():
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
    
    s1_norms = s1_images / (np.linalg.norm(s1_images, axis=1, keepdims=True) + 1e-9)
    s2_norms = s2_images / (np.linalg.norm(s2_images, axis=1, keepdims=True) + 1e-9)
    
    sims = s1_norms @ s2_norms.T
    
    diagonal = np.diag(sims)
    print(f"Raw similarity statistics of the diagonal (self-similarity):")
    print(f"  Mean: {np.mean(diagonal):.4f}")
    print(f"  Min: {np.min(diagonal):.4f}")
    print(f"  Max: {np.max(diagonal):.4f}")
    print(f"  Std: {np.std(diagonal):.4f}")
    
    # Let's count how many times the diagonal is NOT the argmax
    mismatches = 0
    for i in range(n_subjects):
        best_match_idx = np.argmax(sims[i, :])
        if best_match_idx != i:
            mismatches += 1
            if mismatches <= 10:
                print(f"  Subj {i+1}: Best match is {best_match_idx+1} (sim: {sims[i, best_match_idx]:.4f}), self-sim: {sims[i, i]:.4f}")
                
    print(f"Number of subjects where diagonal is NOT the argmax (without zero-mean): {mismatches}")

if __name__ == "__main__":
    test_raw_sim()
