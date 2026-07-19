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

import argparse

def main():
    parser = argparse.ArgumentParser(description="Run Gabor+XPCA Benchmark")
    parser.add_argument("--max_components", type=int, default=512, help="Maximum number of PCA components to extract and analyze.")
    args = parser.parse_args()
    max_comp = args.max_components
    
    print("=" * 80)
    print(f"RUNNING BENCHMARK: GABOR PREPROCESSING + XPCA DIAGONAL CALIBRATION (MAX={max_comp})")
    print("=" * 80)
    
    # 0. Setup directories
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    # Define 5 seeds (42 + 4 additional seeds)
    seeds = [42, 43, 44, 45, 46]
    records = []
    
    # Subspace dimensions to evaluate
    base_dimensions = [32, 64, 128, 256, 512]
    dimensions = [d for d in base_dimensions if d <= max_comp]
    if max_comp not in dimensions:
        dimensions.append(max_comp)
    dimensions.sort()
    
    for seed in seeds:
        print("\n" + "-" * 50)
        print(f"Executing experiments for Seed: {seed}")
        print("-" * 50)
        
        # 1. Load raw datasets (zero-mean normalized for baseline PCA)
        X_train_raw, y_train, X_test_raw, y_test, n_classes, _ = load_iitd_dataset(
            data_dir="data/IITD", max_subjects=None, target_size=(128, 128), zero_mean_samples=True, random_split=True, seed=seed
        )
        
        # Load raw datasets without zero-mean (for Gabor filtering)
        X_train_no_zm, _, X_test_no_zm, _, _, _ = load_iitd_dataset(
            data_dir="data/IITD", max_subjects=None, target_size=(128, 128), zero_mean_samples=False, random_split=True, seed=seed
        )
        
        # 2. Extract Gabor features
        X_train_gabor = extract_gabor_features(
            X_train_no_zm, target_size=(128, 128), ksize=15, sigma=5.0, lambd=10.0, gamma=0.5,
            downsample_size=(52, 52), orientations=[0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]
        )
        X_test_gabor = extract_gabor_features(
            X_test_no_zm, target_size=(128, 128), ksize=15, sigma=5.0, lambd=10.0, gamma=0.5,
            downsample_size=(52, 52), orientations=[0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]
        )
        
        # Center Gabor features to zero mean individually (same as raw pixel pipeline)
        X_train_gabor = X_train_gabor - np.mean(X_train_gabor, axis=1, keepdims=True)
        X_test_gabor = X_test_gabor - np.mean(X_test_gabor, axis=1, keepdims=True)
        
        # 3. Fit baseline PCAs (max_comp components)
        print(f"\nFitting PCA on raw pixels (n={max_comp})...")
        pca_raw = PCA(n_components=max_comp, random_state=seed)
        pca_raw.fit(X_train_raw)
        
        print(f"Fitting PCA on Gabor features (n={max_comp})...")
        pca_gabor = PCA(n_components=max_comp, random_state=seed)
        pca_gabor.fit(X_train_gabor)
        
        # Pre-compute Sort mode on the full max_comp-dimensional Gabor space
        print(f"\nPre-computing XPCA (Sort) on full {max_comp} Gabor dimensions for Feature Selection...")
        tr_gabor_full_max = pca_gabor.transform(X_train_gabor)
        te_gabor_full_max = pca_gabor.transform(X_test_gabor)
        xpca_sort_full = XPCACalibration(scale_method="sort", bootstrap_iter=100, whitening_power=0.2)
        xpca_sort_full.fit(tr_gabor_full_max, y_train, pca_gabor.explained_variance_)
        tr_gabor_sort_full = xpca_sort_full.transform(tr_gabor_full_max)
        te_gabor_sort_full = xpca_sort_full.transform(te_gabor_full_max)
        
        print(f"Pre-computing XPCA (Sort Weights) on full {max_comp} Gabor dimensions...")
        xpca_sort_w_full = XPCACalibration(scale_method="sort_weights", bootstrap_iter=100, whitening_power=0.2)
        xpca_sort_w_full.fit(tr_gabor_full_max, y_train, pca_gabor.explained_variance_)
        tr_gabor_sort_w_full = xpca_sort_w_full.transform(tr_gabor_full_max)
        te_gabor_sort_w_full = xpca_sort_w_full.transform(te_gabor_full_max)
        
        # 4. Comparative loop across dimensions
        for k in dimensions:
            print(f"Evaluating Subspace Dimension k = {k}...")
            
            # Projections
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
            eer_raw_pca = res_raw_pca["eer"] * 100
            
            # 1b. Raw XPCA (Min-Max)
            xpca_raw_mm = XPCACalibration(scale_method="min-max", bootstrap_iter=100, whitening_power=0.2)
            xpca_raw_mm.fit(tr_raw, y_train, pca_raw.explained_variance_[:k])
            tr_raw_mm = xpca_raw_mm.transform(tr_raw)
            te_raw_mm = xpca_raw_mm.transform(te_raw)
            res_raw_xpca = evaluate_verification(tr_raw_mm, y_train, te_raw_mm, y_test, n_classes)
            eer_raw_xpca = res_raw_xpca["eer"] * 100
            
            # ==========================================
            # 2. GABOR PREPROCESSED METHODS
            # ==========================================
            # 2a. Gabor + PCA (No Calibration)
            res_gab_pca = evaluate_verification(tr_gabor, y_train, te_gabor, y_test, n_classes)
            eer_gab_pca = res_gab_pca["eer"] * 100
            
            # 2b. Gabor + XPCA (Min-Max Calibration)
            xpca_gab_mm = XPCACalibration(scale_method="min-max", bootstrap_iter=100, whitening_power=0.2)
            xpca_gab_mm.fit(tr_gabor, y_train, pca_gabor.explained_variance_[:k])
            tr_gab_mm = xpca_gab_mm.transform(tr_gabor)
            te_gab_mm = xpca_gab_mm.transform(te_gabor)
            res_gab_xpca_mm = evaluate_verification(tr_gab_mm, y_train, te_gab_mm, y_test, n_classes)
            eer_gab_xpca_mm = res_gab_xpca_mm["eer"] * 100
            
            # 2c. Gabor + XPCA (Softmax Scaling)
            xpca_gab_sm = XPCACalibration(scale_method="softmax", temp=0.5, bootstrap_iter=100, whitening_power=0.2)
            xpca_gab_sm.fit(tr_gabor, y_train, pca_gabor.explained_variance_[:k])
            tr_gab_sm = xpca_gab_sm.transform(tr_gabor)
            te_gab_sm = xpca_gab_sm.transform(te_gabor)
            res_gab_xpca_sm = evaluate_verification(tr_gab_sm, y_train, te_gab_sm, y_test, n_classes)
            eer_gab_xpca_sm = res_gab_xpca_sm["eer"] * 100
            
            # 2d. Gabor + XPCA (Residual Scaling)
            xpca_gab_res = XPCACalibration(scale_method="residual", eta=0.2, bootstrap_iter=100, whitening_power=0.2)
            xpca_gab_res.fit(tr_gabor, y_train, pca_gabor.explained_variance_[:k])
            tr_gab_res = xpca_gab_res.transform(tr_gabor)
            te_gab_res = xpca_gab_res.transform(te_gabor)
            res_gab_xpca_res = evaluate_verification(tr_gab_res, y_train, te_gab_res, y_test, n_classes)
            eer_gab_xpca_res = res_gab_xpca_res["eer"] * 100
            
            # 2e. Gabor + XPCA (Sort Mode)
            tr_gab_sort = tr_gabor_sort_full[:, :k]
            te_gab_sort = te_gabor_sort_full[:, :k]
            res_gab_xpca_sort = evaluate_verification(tr_gab_sort, y_train, te_gab_sort, y_test, n_classes)
            eer_gab_xpca_sort = res_gab_xpca_sort["eer"] * 100
            
            # 2f. Gabor + XPCA (Sort Weights)
            tr_gab_sort_w = tr_gabor_sort_w_full[:, :k]
            te_gab_sort_w = te_gabor_sort_w_full[:, :k]
            res_gab_xpca_sort_w = evaluate_verification(tr_gab_sort_w, y_train, te_gab_sort_w, y_test, n_classes)
            eer_gab_xpca_sort_w = res_gab_xpca_sort_w["eer"] * 100
            
            # Append evaluations
            methods_results = [
                ("Raw PCA", res_raw_pca, 0.0),
                ("Raw XPCA (Min-Max)", res_raw_xpca, eer_raw_xpca - eer_raw_pca),
                ("Gabor + PCA", res_gab_pca, 0.0),
                ("Gabor + XPCA (Min-Max)", res_gab_xpca_mm, eer_gab_xpca_mm - eer_gab_pca),
                ("Gabor + XPCA (Softmax)", res_gab_xpca_sm, eer_gab_xpca_sm - eer_gab_pca),
                ("Gabor + XPCA (Residual)", res_gab_xpca_res, eer_gab_xpca_res - eer_gab_pca),
                ("Gabor + XPCA (Sort)", res_gab_xpca_sort, eer_gab_xpca_sort - eer_gab_pca),
                ("Gabor + XPCA (Sort Weights)", res_gab_xpca_sort_w, eer_gab_xpca_sort_w - eer_gab_pca)
            ]
            
            for method_name, res, d_eer in methods_results:
                records.append({
                    "seed": seed,
                    "dimension": k,
                    "method": method_name,
                    "eer": res["eer"] * 100,
                    "accuracy": res["accuracy"] * 100,
                    "auc": res["auc"],
                    "delta_eer": d_eer
                })

    # Save findings
    df_results = pd.DataFrame(records)
    df_results.to_csv("results/tables/gabor_xpca_comparison_sort_raw.csv", index=False)
    print("\nSaved raw results to results/tables/gabor_xpca_comparison_sort_raw.csv")
    
    # Calculate summary statistics: mean and std for each metric
    summary_df = df_results.groupby(["dimension", "method"]).agg({
        "eer": ["mean", "std"],
        "accuracy": ["mean", "std"],
        "auc": ["mean", "std"],
        "delta_eer": ["mean", "std"]
    }).reset_index()
    
    # Flatten columns
    summary_df.columns = [
        "dimension", "method", 
        "eer_mean", "eer_std", 
        "accuracy_mean", "accuracy_std", 
        "auc_mean", "auc_std", 
        "delta_eer_mean", "delta_eer_std"
    ]
    
    summary_df.to_csv("results/tables/gabor_xpca_comparison_sort.csv", index=False)
    print("Saved aggregated summary to results/tables/gabor_xpca_comparison_sort.csv")
    
    # Print formatted markdown tables for each value of k
    ordered_methods = [
        "Raw PCA", "Raw XPCA (Min-Max)", 
        "Gabor + PCA", 
        "Gabor + XPCA (Min-Max)", "Gabor + XPCA (Softmax)", "Gabor + XPCA (Residual)",
        "Gabor + XPCA (Sort)", "Gabor + XPCA (Sort Weights)"
    ]
    
    print("\n" + "="*85)
    print("BENCHMARK SUMMARY BY DIMENSION k (Averaged over 5 seeds)")
    print("="*85)
    
    for k in sorted(df_results["dimension"].unique()):
        print(f"\n--- Dimension k = {k} ---")
        k_df = summary_df[summary_df["dimension"] == k].copy()
        k_df = k_df.set_index("method").reindex(ordered_methods).reset_index()
        
        table_rows = []
        for _, row in k_df.iterrows():
            method_name = row["method"]
            if pd.isna(row["eer_mean"]):
                continue
            
            # Safely handle std if NaN (e.g., if only one seed was run)
            eer_std = row["eer_std"] if not pd.isna(row["eer_std"]) else 0.0
            acc_std = row["accuracy_std"] if not pd.isna(row["accuracy_std"]) else 0.0
            auc_std = row["auc_std"] if not pd.isna(row["auc_std"]) else 0.0
            d_eer_std = row["delta_eer_std"] if not pd.isna(row["delta_eer_std"]) else 0.0
            
            eer_str = f"{row['eer_mean']:.2f}% +/- {eer_std:.2f}%"
            acc_str = f"{row['accuracy_mean']:.2f}% +/- {acc_std:.2f}%"
            auc_str = f"{row['auc_mean']:.4f} +/- {auc_std:.4f}"
            
            d_eer_mean = row['delta_eer_mean']
            if method_name in ["Raw PCA", "Gabor + PCA"]:
                d_eer_str = "0.00% (Baseline)"
            else:
                sign = "+" if d_eer_mean > 0 else ""
                d_eer_str = f"{sign}{d_eer_mean:.2f}% +/- {d_eer_std:.2f}%"
                
            table_rows.append({
                "Method": method_name,
                "EER": eer_str,
                "Accuracy": acc_str,
                "AUC": auc_str,
                "Delta EER": d_eer_str
            })
            
        print(pd.DataFrame(table_rows).to_markdown(index=False))
        
    print("\n" + "="*85)

if __name__ == "__main__":
    main()
