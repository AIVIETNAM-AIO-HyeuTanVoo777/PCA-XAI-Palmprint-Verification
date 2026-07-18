import os
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing.loader import load_iitd_dataset
from preprocessing.gabor import extract_gabor_features
from explainability.xpca import XPCACalibration
from evaluation.metrics import evaluate_verification

def main():
    print("=" * 80)
    print("REPRODUCING BENCHMARK: GABOR PREPROCESSING + XPCA DIAGONAL CALIBRATION")
    print("=" * 80)
    
    # 1. Load raw datasets (zero-mean normalized for baseline PCA)
    X_train_raw, y_train, X_test_raw, y_test, n_classes, _ = load_iitd_dataset(
        data_dir="data/IITD", max_subjects=None, target_size=(128, 128), zero_mean_samples=True
    )
    
    # Load raw datasets without zero-mean (for Gabor filtering)
    X_train_no_zm, _, X_test_no_zm, _, _, _ = load_iitd_dataset(
        data_dir="data/IITD", max_subjects=None, target_size=(128, 128), zero_mean_samples=False
    )
    
    # 2. Extract Gabor features
    X_train_gabor = extract_gabor_features(X_train_no_zm, target_size=(128, 128), downsample_size=(64, 64))
    X_test_gabor = extract_gabor_features(X_test_no_zm, target_size=(128, 128), downsample_size=(64, 64))
    
    # Center Gabor features to zero mean individually (same as raw pixel pipeline)
    X_train_gabor = X_train_gabor - np.mean(X_train_gabor, axis=1, keepdims=True)
    X_test_gabor = X_test_gabor - np.mean(X_test_gabor, axis=1, keepdims=True)
    
    # 3. Fit baseline PCAs (512 components)
    print("\nFitting PCA on raw pixels...")
    pca_raw = PCA(n_components=512, random_state=42)
    pca_raw.fit(X_train_raw)
    
    print("Fitting PCA on Gabor features...")
    pca_gabor = PCA(n_components=512, random_state=42)
    pca_gabor.fit(X_train_gabor)
    
    # 4. Comparative loop across dimensions
    dimensions = [32, 64, 128, 256, 512]
    records = []
    
    for k in dimensions:
        print(f"\n--- Evaluating Subspace Dimension k = {k} ---")
        
        # --- PREPARE PROJECTIONS ---
        tr_raw_full = pca_raw.transform(X_train_raw)
        te_raw_full = pca_raw.transform(X_test_raw)
        tr_raw = tr_raw_full[:, :k]
        te_raw = te_raw_full[:, :k]
        
        tr_gabor_full = pca_gabor.transform(X_train_gabor)
        te_gabor_full = pca_gabor.transform(X_test_gabor)
        tr_gabor = tr_gabor_full[:, :k]
        te_gabor = te_gabor_full[:, :k]
        
        # ==========================================
        # 1. RAW PIXEL BASELINES
        # ==========================================
        # 1a. Raw PCA (No Calibration)
        res_raw_pca = evaluate_verification(tr_raw, y_train, te_raw, y_test, n_classes)
        records.append({
            "dimension": k,
            "method": "Raw PCA",
            "eer": res_raw_pca["eer"] * 100,
            "accuracy": res_raw_pca["accuracy"] * 100,
            "auc": res_raw_pca["auc"]
        })
        
        # 1b. Raw XPCA (Min-Max)
        xpca_raw_mm = XPCACalibration(scale_method="min-max", bootstrap_iter=100)
        xpca_raw_mm.fit(tr_raw, y_train, pca_raw.explained_variance_[:k])
        tr_raw_mm = xpca_raw_mm.transform(tr_raw)
        te_raw_mm = xpca_raw_mm.transform(te_raw)
        res_raw_xpca = evaluate_verification(tr_raw_mm, y_train, te_raw_mm, y_test, n_classes)
        records.append({
            "dimension": k,
            "method": "Raw XPCA (Min-Max)",
            "eer": res_raw_xpca["eer"] * 100,
            "accuracy": res_raw_xpca["accuracy"] * 100,
            "auc": res_raw_xpca["auc"]
        })
        
        # ==========================================
        # 2. GABOR PREPROCESSED METHODS
        # ==========================================
        # 2a. Gabor + PCA (No Calibration)
        res_gab_pca = evaluate_verification(tr_gabor, y_train, te_gabor, y_test, n_classes)
        records.append({
            "dimension": k,
            "method": "Gabor + PCA",
            "eer": res_gab_pca["eer"] * 100,
            "accuracy": res_gab_pca["accuracy"] * 100,
            "auc": res_gab_pca["auc"]
        })
        
        # 2b. Gabor + XPCA (Min-Max Calibration)
        xpca_gab_mm = XPCACalibration(scale_method="min-max", bootstrap_iter=100)
        xpca_gab_mm.fit(tr_gabor, y_train, pca_gabor.explained_variance_[:k])
        tr_gab_mm = xpca_gab_mm.transform(tr_gabor)
        te_gab_mm = xpca_gab_mm.transform(te_gabor)
        res_gab_xpca_mm = evaluate_verification(tr_gab_mm, y_train, te_gab_mm, y_test, n_classes)
        records.append({
            "dimension": k,
            "method": "Gabor + XPCA (Min-Max)",
            "eer": res_gab_xpca_mm["eer"] * 100,
            "accuracy": res_gab_xpca_mm["accuracy"] * 100,
            "auc": res_gab_xpca_mm["auc"]
        })
        
        # 2c. Gabor + XPCA (Softmax Scaling - Regularized)
        xpca_gab_sm = XPCACalibration(scale_method="softmax", temp=0.5, bootstrap_iter=100)
        xpca_gab_sm.fit(tr_gabor, y_train, pca_gabor.explained_variance_[:k])
        tr_gab_sm = xpca_gab_sm.transform(tr_gabor)
        te_gab_sm = xpca_gab_sm.transform(te_gabor)
        res_gab_xpca_sm = evaluate_verification(tr_gab_sm, y_train, te_gab_sm, y_test, n_classes)
        records.append({
            "dimension": k,
            "method": "Gabor + XPCA (Softmax)",
            "eer": res_gab_xpca_sm["eer"] * 100,
            "accuracy": res_gab_xpca_sm["accuracy"] * 100,
            "auc": res_gab_xpca_sm["auc"]
        })
        
        # 2d. Gabor + XPCA (Residual Scaling - Regularized)
        xpca_gab_res = XPCACalibration(scale_method="residual", eta=0.2, bootstrap_iter=100)
        xpca_gab_res.fit(tr_gabor, y_train, pca_gabor.explained_variance_[:k])
        tr_gab_res = xpca_gab_res.transform(tr_gabor)
        te_gab_res = xpca_gab_res.transform(te_gabor)
        res_gab_xpca_res = evaluate_verification(tr_gab_res, y_train, te_gab_res, y_test, n_classes)
        records.append({
            "dimension": k,
            "method": "Gabor + XPCA (Residual)",
            "eer": res_gab_xpca_res["eer"] * 100,
            "accuracy": res_gab_xpca_res["accuracy"] * 100,
            "auc": res_gab_xpca_res["auc"]
        })
        
    df_results = pd.DataFrame(records)
    df_results.to_csv("_paper_resolution/gabor_xpca_comparison_reproduced.csv", index=False)
    print("\nSaved comprehensive results to _paper_resolution/gabor_xpca_comparison_reproduced.csv")

if __name__ == "__main__":
    main()
