import os
import glob
import cv2
import numpy as np
from sklearn.decomposition import PCA
from verification import compute_eer

def test_whitening(zero_mean_samples=True, use_remapping=True, whiten=True):
    sess1_dir = "Tongji Dataset/session1"
    sess2_dir = "Tongji Dataset/session2"
    
    sess1_files = sorted(glob.glob(os.path.join(sess1_dir, "*.bmp")))
    sess2_files = sorted(glob.glob(os.path.join(sess2_dir, "*.bmp")))
    
    n_subjects = 100
    target_size = (128, 128)
    
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
    
    if zero_mean_samples:
        X_train = X_train - np.mean(X_train, axis=1, keepdims=True)
        X_test = X_test - np.mean(X_test, axis=1, keepdims=True)
        
    # Fit PCA
    pca = PCA(n_components=128, random_state=42).fit(X_train)
    tr_proj = pca.transform(X_train)
    te_proj = pca.transform(X_test)
    
    if whiten:
        # Whitening: divide by sqrt of eigenvalues (which are the explained variances)
        # eigenvalues are stored in pca.explained_variance_
        std_devs = np.sqrt(pca.explained_variance_)
        # clip std_devs to avoid division by zero
        std_devs = np.clip(std_devs, 1e-9, None)
        tr_proj = tr_proj / std_devs
        te_proj = te_proj / std_devs
        
    # Templates (mean train vector)
    templates = []
    for c in range(n_subjects):
        templates.append(np.mean(tr_proj[y_train == c], axis=0))
    templates = np.array(templates)
    
    # Normalize
    te_norm = te_proj / np.clip(np.linalg.norm(te_proj, axis=1, keepdims=True), 1e-9, None)
    temp_norm = templates / np.clip(np.linalg.norm(templates, axis=1, keepdims=True), 1e-9, None)
    
    # Similarity matrix
    sim_matrix = te_norm @ temp_norm.T
    
    # Extract genuine and imposter scores
    genuine_scores = sim_matrix[np.arange(len(y_test)), y_test]
    mask = np.ones_like(sim_matrix, dtype=bool)
    mask[np.arange(len(y_test)), y_test] = False
    imposter_scores = sim_matrix[mask]
    
    eer, th, fpr, fnr = compute_eer(genuine_scores, imposter_scores)
    print(f"PCA EER (whiten={whiten}): {eer:.6f} at threshold {th:.4f}")

if __name__ == "__main__":
    test_whitening(whiten=False)
    test_whitening(whiten=True)
