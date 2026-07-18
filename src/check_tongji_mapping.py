import os
import glob
import cv2
import numpy as np

def analyze_mapping():
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
    
    # Calculate similarity
    sims = s1_norms @ s2_norms.T # shape: (n_subjects, n_subjects)
    
    # For each Session 2 subject, find the best Session 1 subject
    s2_to_s1_map = {}
    s1_to_s2_map = {}
    
    mismatches = 0
    correct_matches = 0
    
    for s2_idx in range(n_subjects):
        # sims[:, s2_idx] gets the similarity of all s1 subjects with this s2 subject
        best_s1_idx = np.argmax(sims[:, s2_idx])
        max_sim = sims[best_s1_idx, s2_idx]
        self_sim = sims[s2_idx, s2_idx]
        
        s2_to_s1_map[s2_idx] = (best_s1_idx, max_sim, self_sim)
        
        if best_s1_idx == s2_idx:
            correct_matches += 1
        else:
            mismatches += 1
            if mismatches <= 20:
                print(f"Sess2 Subj {s2_idx+1} (self-sim: {self_sim:.4f}) -> Best Sess1 Subj {best_s1_idx+1} (sim: {max_sim:.4f})")
                
    print(f"\nSummary of mappings using argmax:")
    print(f"Total subjects: {n_subjects}")
    print(f"Perfect diagonal matches (argmax(s1) == s2): {correct_matches}")
    print(f"Mismatches: {mismatches}")
    
    # Let's see if the mismatches are a permutation or shift
    # For example, does Session 2 subject 1 match Session 1 subject 321, and what about subject 2, etc?
    # Let's count how many times each Session 1 subject is mapped to
    mapped_s1_counts = {}
    for s2_idx, (best_s1_idx, _, _) in s2_to_s1_map.items():
        mapped_s1_counts[best_s1_idx] = mapped_s1_counts.get(best_s1_idx, 0) + 1
        
    print(f"Unique Session 1 subjects mapped to: {len(mapped_s1_counts)}")
    duplicate_mappings = {k: v for k, v in mapped_s1_counts.items() if v > 1}
    print(f"Duplicate mappings (multiple Sess2 subjects mapping to same Sess1 subject): {len(duplicate_mappings)}")
    
if __name__ == "__main__":
    analyze_mapping()
