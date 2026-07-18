import os
import numpy as np
from sklearn.decomposition import PCA
from dataset_loader import load_tongji_dataset

def check_indiv():
    # Load Tongji dataset without remapping to see raw similarities
    # We load 100 subjects
    # Let's temporarily disable the mapping file by renaming it if it exists
    mapping_exists = os.path.exists("Tongji_session2_mapping.csv")
    if mapping_exists:
        os.rename("Tongji_session2_mapping.csv", "Tongji_session2_mapping_temp.csv")
        
    try:
        X_tr, y_tr, X_te, y_te, n_classes = load_tongji_dataset("data/Tongji", max_subjects=100)
    finally:
        if mapping_exists:
            os.rename("Tongji_session2_mapping_temp.csv", "Tongji_session2_mapping.csv")
            
    pca = PCA(n_components=128, random_state=42).fit(X_tr)
    tr_proj = pca.transform(X_tr)
    te_proj = pca.transform(X_te)
    
    # Compute templates
    templates = []
    for c in range(n_classes):
        templates.append(np.mean(tr_proj[y_tr == c], axis=0))
    templates = np.array(templates)
    
    # Normalize
    te_norm = te_proj / np.clip(np.linalg.norm(te_proj, axis=1, keepdims=True), 1e-9, None)
    temp_norm = templates / np.clip(np.linalg.norm(templates, axis=1, keepdims=True), 1e-9, None)
    
    sim_matrix = te_norm @ temp_norm.T # shape: (1000, 100)
    
    print("\nSubject-wise verification details for first 15 subjects:")
    for c in range(15):
        # Average similarity of subject c's 10 test images with all templates
        sub_sims = sim_matrix[c*10 : (c+1)*10] # shape: (10, 100)
        mean_sims = np.mean(sub_sims, axis=0) # shape: (100,)
        
        self_sim = mean_sims[c]
        best_match_idx = np.argmax(mean_sims)
        best_sim = mean_sims[best_match_idx]
        
        print(f"  Subject {c+1}: Self-sim = {self_sim:.4f}, Best match = Subject {best_match_idx+1} ({best_sim:.4f})")
        if best_match_idx != c:
            print(f"    *** MISMATCH: Subject {c+1} matches Subject {best_match_idx+1} better! ***")

if __name__ == "__main__":
    check_indiv()
