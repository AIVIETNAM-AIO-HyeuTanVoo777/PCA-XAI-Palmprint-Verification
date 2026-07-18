import os
import glob
import cv2
import numpy as np
from verification import compute_eer

def evaluate_pixel_space(use_remapping=False):
    print(f"Evaluating in pixel space (use_remapping={use_remapping})...")
    sess1_dir = "Tongji Dataset/session1"
    sess2_dir = "Tongji Dataset/session2"
    
    sess1_files = sorted(glob.glob(os.path.join(sess1_dir, "*.bmp")))
    sess2_files = sorted(glob.glob(os.path.join(sess2_dir, "*.bmp")))
    
    # Let's load first 100 subjects
    n_subjects = 100
    target_size = (64, 64)
    
    # Load mapping
    s1_to_s2 = {i: i for i in range(600)}
    if use_remapping and os.path.exists("Tongji_session2_mapping.csv"):
        import pandas as pd
        df_map = pd.read_csv("Tongji_session2_mapping.csv")
        s1_to_s2 = dict(zip(df_map['session1_index'], df_map['session2_index']))
        
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    for subj_idx in range(n_subjects):
        # Train (Session 1)
        for f in sess1_files[subj_idx * 10 : (subj_idx + 1) * 10]:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_res = cv2.resize(img, target_size).astype(np.float32) / 255.0
                X_train.append(img_res.flatten())
                y_train.append(subj_idx)
                
        # Test (Session 2)
        s2_idx = s1_to_s2.get(subj_idx, subj_idx)
        for f in sess2_files[s2_idx * 10 : (s2_idx + 1) * 10]:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_res = cv2.resize(img, target_size).astype(np.float32) / 255.0
                X_test.append(img_res.flatten())
                y_test.append(subj_idx)
                
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    # Zero-mean normalize each sample
    X_train_zm = X_train - np.mean(X_train, axis=1, keepdims=True)
    X_test_zm = X_test - np.mean(X_test, axis=1, keepdims=True)
    
    # Norm
    X_train_norm = X_train_zm / np.clip(np.linalg.norm(X_train_zm, axis=1, keepdims=True), 1e-9, None)
    X_test_norm = X_test_zm / np.clip(np.linalg.norm(X_test_zm, axis=1, keepdims=True), 1e-9, None)
    
    # Templates (mean train vector)
    templates = []
    for c in range(n_subjects):
        templates.append(np.mean(X_train_norm[y_train == c], axis=0))
    templates = np.array(templates)
    templates_norm = templates / np.clip(np.linalg.norm(templates, axis=1, keepdims=True), 1e-9, None)
    
    # Similarity matrix
    sim_matrix = X_test_norm @ templates_norm.T
    
    # Extract genuine and imposter scores
    genuine_scores = sim_matrix[np.arange(len(y_test)), y_test]
    mask = np.ones_like(sim_matrix, dtype=bool)
    mask[np.arange(len(y_test)), y_test] = False
    imposter_scores = sim_matrix[mask]
    
    eer, th, fpr, fnr = compute_eer(genuine_scores, imposter_scores)
    print(f"EER in pixel space: {eer:.6f} at threshold {th:.4f}")
    print(f"Mean Genuine Similarity: {np.mean(genuine_scores):.4f}")
    print(f"Mean Imposter Similarity: {np.mean(imposter_scores):.4f}")

if __name__ == "__main__":
    evaluate_pixel_space(use_remapping=False)
    evaluate_pixel_space(use_remapping=True)
