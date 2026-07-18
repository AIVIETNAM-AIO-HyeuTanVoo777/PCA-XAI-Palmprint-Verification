import os
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing.loader import load_iitd_dataset
from preprocessing.gabor import extract_gabor_features
from evaluation.metrics import evaluate_verification

def main():
    print("=" * 85)
    print("RUNNING COMPREHENSIVE BENCHMARK: EER (%), Acc. (%), AUC, and Delta EER")
    print("=" * 85)
    
    # 1. Load datasets
    X_train_raw, y_train, X_test_raw, y_test, n_classes, _ = load_iitd_dataset(
        data_dir="data/IITD", max_subjects=None, target_size=(128, 128), zero_mean_samples=True
    )
    
    X_train_no_zm, _, X_test_no_zm, _, _, _ = load_iitd_dataset(
        data_dir="data/IITD", max_subjects=None, target_size=(128, 128), zero_mean_samples=False
    )
    
    # Extract Gabor features using optimal parameters (6 dirs, lambd=10.0)
    print("\nExtracting Gabor features with optimal params (6 orientations, lambd=10.0)...")
    X_train_gabor = extract_gabor_features(
        X_train_no_zm, target_size=(128, 128), ksize=15, sigma=5.0, lambd=10.0, gamma=0.5,
        downsample_size=(52, 52), orientations=[0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]
    )
    X_test_gabor = extract_gabor_features(
        X_test_no_zm, target_size=(128, 128), ksize=15, sigma=5.0, lambd=10.0, gamma=0.5,
        downsample_size=(52, 52), orientations=[0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]
    )
    
    # Individual zero-mean normalization
    X_train_gabor = X_train_gabor - np.mean(X_train_gabor, axis=1, keepdims=True)
    X_test_gabor = X_test_gabor - np.mean(X_test_gabor, axis=1, keepdims=True)
    
    # 2. Fit full PCAs
    print("\nFitting PCAs...")
    pca_raw = PCA(n_components=512, random_state=42)
    tr_raw_full = pca_raw.fit_transform(X_train_raw)
    te_raw_full = pca_raw.transform(X_test_raw)
    
    pca_gabor = PCA(n_components=512, random_state=42)
    tr_gab_full = pca_gabor.fit_transform(X_train_gabor)
    te_gab_full = pca_gabor.transform(X_test_gabor)
    
    # 3. Setup optimal XPCA hyperparams
    opt_sh = 0.7
    opt_w = (0.0, 0.8, 0.2)
    opt_eta = 0.1
    
    # Standard XPCA hyperparams (Min-Max)
    std_sh = 0.1
    std_w = (0.2, 0.6, 0.2)
    
    dimensions = [32, 64, 128, 256, 512]
    records = []
    
    eigenvalues_raw = pca_raw.explained_variance_
    eigenvalues_gab = pca_gabor.explained_variance_
    n_samples = len(y_train)
    
    for k in dimensions:
        print(f"Evaluating k = {k}...")
        tr_raw = tr_raw_full[:, :k]
        te_raw = te_raw_full[:, :k]
        
        tr_gab = tr_gab_full[:, :k]
        te_gab = te_gab_full[:, :k]
        
        # ----------------------------------------------------
        # Method 1: Raw PCA (Baseline)
        # ----------------------------------------------------
        res_raw_pca = evaluate_verification(tr_raw, y_train, te_raw, y_test, n_classes)
        eer_raw_pca = res_raw_pca["eer"] * 100
        records.append({
            "dimension": k,
            "method": "Raw PCA",
            "eer": eer_raw_pca,
            "accuracy": res_raw_pca["accuracy"] * 100,
            "auc": res_raw_pca["auc"],
            "delta_eer": 0.0
        })
        
        # ----------------------------------------------------
        # Method 2: Raw XPCA (Min-Max)
        # ----------------------------------------------------
        # Compute Fisher & Bootstrap for Raw XPCA
        classes = np.unique(y_train)
        n_cl = len(classes)
        class_means = np.zeros((n_cl, k))
        class_vars = np.zeros((n_cl, k))
        for idx, c in enumerate(classes):
            mask = (y_train == c)
            z_c = tr_raw[mask]
            if len(z_c) > 0:
                class_means[idx] = np.mean(z_c, axis=0)
                if len(z_c) > 1:
                    class_vars[idx] = np.var(z_c, axis=0, ddof=1)
                    
        S_B = np.var(class_means, axis=0)
        S_W = (1.0 - std_sh) * np.mean(class_vars, axis=0) + std_sh * eigenvalues_raw[:k]
        D_raw = S_B / (S_W + 1e-10)
        
        rng = np.random.default_rng(42)
        bootstrap_fisher = np.zeros((30, k))
        for t in range(30):
            indices = rng.choice(n_samples, n_samples, replace=True)
            z_res = tr_raw[indices]
            y_res = y_train[indices]
            classes_t, counts_t = np.unique(y_res, return_counts=True)
            res_means = []
            res_vars = []
            for c, count in zip(classes_t, counts_t):
                z_c = z_res[y_res == c]
                res_means.append(np.mean(z_c, axis=0))
                if count > 1:
                    res_vars.append(np.var(z_c, axis=0, ddof=1))
                else:
                    res_vars.append(np.zeros(k))
            if len(res_means) > 0:
                res_means = np.array(res_means)
                res_vars = np.array(res_vars)
                S_B_t = np.var(res_means, axis=0)
                S_W_t = (1.0 - std_sh) * np.mean(res_vars, axis=0) + std_sh * eigenvalues_raw[:k]
                bootstrap_fisher[t] = S_B_t / (S_W_t + 1e-10)
                
        N_raw = np.std(bootstrap_fisher, axis=0) / (np.mean(bootstrap_fisher, axis=0) + 1e-10)
        V_raw = eigenvalues_raw[:k] / np.sum(eigenvalues_raw)
        
        raw_util = std_w[0]*V_raw + std_w[1]*D_raw - std_w[2]*N_raw
        min_v = np.min(raw_util)
        max_v = np.max(raw_util)
        weights_raw = (raw_util - min_v) / (max_v - min_v + 1e-10) # Min-Max
        
        tr_raw_xpca = tr_raw * weights_raw
        te_raw_xpca = te_raw * weights_raw
        res_raw_xpca = evaluate_verification(tr_raw_xpca, y_train, te_raw_xpca, y_test, n_classes)
        records.append({
            "dimension": k,
            "method": "Raw XPCA (Min-Max)",
            "eer": res_raw_xpca["eer"] * 100,
            "accuracy": res_raw_xpca["accuracy"] * 100,
            "auc": res_raw_xpca["auc"],
            "delta_eer": (res_raw_xpca["eer"] * 100) - eer_raw_pca
        })
        
        # ----------------------------------------------------
        # Method 3: Gabor + PCA
        # ----------------------------------------------------
        res_gab_pca = evaluate_verification(tr_gab, y_train, te_gab, y_test, n_classes)
        eer_gab_pca = res_gab_pca["eer"] * 100
        records.append({
            "dimension": k,
            "method": "Gabor + PCA",
            "eer": eer_gab_pca,
            "accuracy": res_gab_pca["accuracy"] * 100,
            "auc": res_gab_pca["auc"],
            "delta_eer": 0.0
        })
        
        # ----------------------------------------------------
        # Method 4: Gabor + XPCA (Optimized Residual)
        # ----------------------------------------------------
        class_means = np.zeros((n_cl, k))
        class_vars = np.zeros((n_cl, k))
        for idx, c in enumerate(classes):
            mask = (y_train == c)
            z_c = tr_gab[mask]
            if len(z_c) > 0:
                class_means[idx] = np.mean(z_c, axis=0)
                if len(z_c) > 1:
                    class_vars[idx] = np.var(z_c, axis=0, ddof=1)
                    
        S_B = np.var(class_means, axis=0)
        S_W = (1.0 - opt_sh) * np.mean(class_vars, axis=0) + opt_sh * eigenvalues_gab[:k]
        D_gab = S_B / (S_W + 1e-10)
        
        rng = np.random.default_rng(42)
        bootstrap_fisher = np.zeros((30, k))
        for t in range(30):
            indices = rng.choice(n_samples, n_samples, replace=True)
            z_res = tr_gab[indices]
            y_res = y_train[indices]
            classes_t, counts_t = np.unique(y_res, return_counts=True)
            res_means = []
            res_vars = []
            for c, count in zip(classes_t, counts_t):
                z_c = z_res[y_res == c]
                res_means.append(np.mean(z_c, axis=0))
                if count > 1:
                    res_vars.append(np.var(z_c, axis=0, ddof=1))
                else:
                    res_vars.append(np.zeros(k))
            if len(res_means) > 0:
                res_means = np.array(res_means)
                res_vars = np.array(res_vars)
                S_B_t = np.var(res_means, axis=0)
                S_W_t = (1.0 - opt_sh) * np.mean(res_vars, axis=0) + opt_sh * eigenvalues_gab[:k]
                bootstrap_fisher[t] = S_B_t / (S_W_t + 1e-10)
                
        N_gab = np.std(bootstrap_fisher, axis=0) / (np.mean(bootstrap_fisher, axis=0) + 1e-10)
        V_gab = eigenvalues_gab[:k] / np.sum(eigenvalues_gab)
        
        raw_util_gab = opt_w[0]*V_gab + opt_w[1]*D_gab - opt_w[2]*N_gab
        min_v = np.min(raw_util_gab)
        max_v = np.max(raw_util_gab)
        norm_util_gab = (raw_util_gab - min_v) / (max_v - min_v + 1e-10)
        
        weights_gab = 1.0 + opt_eta * norm_util_gab
        
        tr_gab_xpca = tr_gab * weights_gab
        te_gab_xpca = te_gab * weights_gab
        res_gab_xpca = evaluate_verification(tr_gab_xpca, y_train, te_gab_xpca, y_test, n_classes)
        records.append({
            "dimension": k,
            "method": "Gabor + XPCA (Optimized)",
            "eer": res_gab_xpca["eer"] * 100,
            "accuracy": res_gab_xpca["accuracy"] * 100,
            "auc": res_gab_xpca["auc"],
            "delta_eer": (res_gab_xpca["eer"] * 100) - eer_gab_pca
        })

    # Save to CSV
    df = pd.DataFrame(records)
    df.to_csv("results/tables/comprehensive_benchmark.csv", index=False)
    print("\nSaved comprehensive results to results/tables/comprehensive_benchmark.csv")
    
    # Construct structured output tables
    print("\n" + "=" * 90)
    print("COMPREHENSIVE BENCHMARK EVALUATION TABLE")
    print("=" * 90)
    print(df.to_markdown(index=False, floatfmt=(".0f", "s", ".4f", ".4f", ".4f", ".4f")))
    print("=" * 90)

if __name__ == "__main__":
    main()
