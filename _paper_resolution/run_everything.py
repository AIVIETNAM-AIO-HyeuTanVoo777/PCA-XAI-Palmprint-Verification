import os
import sys
import glob
import cv2
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
import scipy.stats as stats
import time

# Create directories
os.makedirs("_paper_resolution/configs", exist_ok=True)
os.makedirs("_paper_resolution/splits", exist_ok=True)
os.makedirs("_paper_resolution/raw_scores", exist_ok=True)
os.makedirs("_paper_resolution/logs", exist_ok=True)

# Logger setup
log_file = open("_paper_resolution/logs/run_output.log", "w")
def log_print(msg):
    print(msg)
    log_file.write(msg + "\n")
    log_file.flush()

# ---------------------------------------------------------
# Part 1: Loader
# ---------------------------------------------------------
def load_all_iitd(data_dir="data/IITD", target_size=(128, 128)):
    log_print(f"Loading all IITD images from {data_dir}...")
    seg_dir = os.path.join(data_dir, "Segmented")
    left_dir = os.path.join(seg_dir, "Left")
    right_dir = os.path.join(seg_dir, "Right")
    
    if not os.path.exists(left_dir) or not os.path.exists(right_dir):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        left_dir = os.path.join(base_dir, "IITD Dataset", "Segmented", "Left")
        right_dir = os.path.join(base_dir, "IITD Dataset", "Segmented", "Right")
        if not os.path.exists(left_dir) or not os.path.exists(right_dir):
            raise FileNotFoundError(f"Directories not found: {left_dir}")
            
    left_files = glob.glob(os.path.join(left_dir, "*.bmp"))
    right_files = glob.glob(os.path.join(right_dir, "*.bmp"))
    
    def group_by_palm(files, side_label):
        groups = {}
        for f in files:
            basename = os.path.basename(f)
            parts = basename.split("_")
            if len(parts) < 2:
                continue
            subj_id = int(parts[0])
            key = f"{subj_id}_{side_label}"
            if key not in groups:
                groups[key] = []
            groups[key].append(f)
        return groups

    left_groups = group_by_palm(left_files, "L")
    right_groups = group_by_palm(right_files, "R")
    
    all_groups = {}
    all_groups.update(left_groups)
    all_groups.update(right_groups)
    
    sorted_keys = sorted(all_groups.keys(), key=lambda x: (int(x.split("_")[0]), x.split("_")[1]))
    class_map = {k: i for i, k in enumerate(sorted_keys)}
    
    # We will load images in a dictionary structured by class_idx
    loaded_data = {}
    
    for palm_key in sorted_keys:
        files = all_groups[palm_key]
        c_idx = class_map[palm_key]
        
        def get_sample_id(f):
            basename = os.path.basename(f)
            parts = basename.split("_")
            if len(parts) >= 2:
                try:
                    return int(parts[1].split(".")[0])
                except ValueError:
                    pass
            return 999
            
        files_sorted = sorted(files, key=get_sample_id)
        
        class_samples = []
        for f in files_sorted:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            img_flat = img_resized.astype(np.float32) / 255.0
            class_samples.append({
                "filename": os.path.basename(f),
                "filepath": f,
                "img_no_zm": img_flat,
                "img_zm": img_flat - np.mean(img_flat)
            })
        loaded_data[c_idx] = class_samples
        
    log_print(f"Loaded data for {len(loaded_data)} classes.")
    return loaded_data

# ---------------------------------------------------------
# Part 2: Gabor Extract
# ---------------------------------------------------------
def get_gabor_kernels(ksize=15, sigma=4.0, lambd=8.0, gamma=0.5, orientations=None):
    if orientations is None:
        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    kernels = []
    for theta in orientations:
        kernel = cv2.getGaborKernel(
            ksize=(ksize, ksize),
            sigma=sigma,
            theta=theta,
            lambd=lambd,
            gamma=gamma,
            psi=0,
            ktype=cv2.CV_32F
        )
        kernel -= np.mean(kernel)
        kernels.append(kernel)
    return kernels

def extract_gabor_for_samples(class_samples, kernels, downsample_size):
    # Extracts Gabor features for all samples in a list
    gabor_features = []
    for sample in class_samples:
        img_2d = sample["img_no_zm"]
        filtered_list = []
        for kernel in kernels:
            filtered = cv2.filter2D(img_2d, cv2.CV_32F, kernel)
            filtered_mag = np.abs(filtered)
            filtered_resized = cv2.resize(filtered_mag, downsample_size, interpolation=cv2.INTER_LINEAR)
            filtered_list.append(filtered_resized.flatten())
        feat = np.concatenate(filtered_list)
        # Center Gabor features individually
        feat_zm = feat - np.mean(feat)
        sample_copy = sample.copy()
        sample_copy["gabor_zm"] = feat_zm
        gabor_features.append(sample_copy)
    return gabor_features

# ---------------------------------------------------------
# Part 3: EER and Verification
# ---------------------------------------------------------
def compute_eer(genuine_scores, imposter_scores):
    y_true = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(imposter_scores)])
    y_scores = np.concatenate([genuine_scores, imposter_scores])
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    threshold = thresholds[idx]
    return eer, threshold

def evaluate_verification(X_train, y_train, X_test, y_test, n_classes):
    k = X_train.shape[1]
    templates = []
    for c in range(n_classes):
        class_mask = (y_train == c)
        if np.sum(class_mask) == 0:
            templates.append(np.zeros(k))
        else:
            templates.append(np.mean(X_train[class_mask], axis=0))
    templates = np.array(templates)
    
    test_norm = X_test / np.clip(np.linalg.norm(X_test, axis=1, keepdims=True), 1e-9, None)
    temp_norm = templates / np.clip(np.linalg.norm(templates, axis=1, keepdims=True), 1e-9, None)
    
    sim_matrix = test_norm @ temp_norm.T
    
    genuine_scores = sim_matrix[np.arange(len(y_test)), y_test]
    mask = np.ones_like(sim_matrix, dtype=bool)
    mask[np.arange(len(y_test)), y_test] = False
    imposter_scores = sim_matrix[mask]
    
    eer, threshold = compute_eer(genuine_scores, imposter_scores)
    
    y_true = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(imposter_scores)])
    y_scores = np.concatenate([genuine_scores, imposter_scores])
    fpr_auc, tpr_auc, _ = roc_curve(y_true, y_scores, pos_label=1)
    roc_auc = auc(fpr_auc, tpr_auc)
    
    return eer, roc_auc, threshold, genuine_scores, imposter_scores

# ---------------------------------------------------------
# Part 4: XPCA Calibration fit/transform
# ---------------------------------------------------------
def get_xpca_weights(X_train, y_train, eigenvalues, shrinkage=0.1, alpha=0.2, beta=0.6, gamma=0.2, bootstrap_iter=30, seed=42):
    k = X_train.shape[1]
    n_samples = len(y_train)
    classes = np.unique(y_train)
    n_cl = len(classes)
    
    # 1. Variance Score
    V_scores = eigenvalues[:k] / np.sum(eigenvalues)
    
    # 2. Fisher Discriminability Score
    class_means = np.zeros((n_cl, k))
    class_vars = np.zeros((n_cl, k))
    for idx, c in enumerate(classes):
        mask = (y_train == c)
        z_c = X_train[mask]
        if len(z_c) > 0:
            class_means[idx] = np.mean(z_c, axis=0)
            if len(z_c) > 1:
                class_vars[idx] = np.var(z_c, axis=0, ddof=1)
    S_B = np.var(class_means, axis=0)
    S_W = (1.0 - shrinkage) * np.mean(class_vars, axis=0) + shrinkage * eigenvalues[:k]
    D_scores = S_B / (S_W + 1e-10)
    
    # 3. Bootstrap Instability Score
    rng = np.random.default_rng(seed)
    bootstrap_fisher = np.zeros((bootstrap_iter, k))
    for t in range(bootstrap_iter):
        indices = rng.choice(n_samples, n_samples, replace=True)
        z_res = X_train[indices]
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
            S_W_t = (1.0 - shrinkage) * np.mean(res_vars, axis=0) + shrinkage * eigenvalues[:k]
            bootstrap_fisher[t] = S_B_t / (S_W_t + 1e-10)
            
    N_scores = np.std(bootstrap_fisher, axis=0) / (np.mean(bootstrap_fisher, axis=0) + 1e-10)
    
    raw_utility = alpha * V_scores + beta * D_scores - gamma * N_scores
    min_val = np.min(raw_utility)
    max_val = np.max(raw_utility)
    norm_utility = (raw_utility - min_val) / (max_val - min_val + 1e-10)
    return norm_utility

# ---------------------------------------------------------
# Part 5: Partitioning and Trial Executor
# ---------------------------------------------------------
def get_split_data(loaded_data, seed=None):
    # If seed is None, use deterministic split (first 3 sorted images)
    # If seed is provided, use random enrollment/probe split
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    filenames_train = []
    filenames_test = []
    
    n_classes = len(loaded_data)
    for c_idx in range(n_classes):
        samples = loaded_data[c_idx]
        if seed is None:
            train_samples = samples[:3]
            test_samples = samples[3:]
        else:
            # Replicates random split from loader.py
            rng = np.random.default_rng(seed + c_idx)
            shuffled = samples.copy()
            rng.shuffle(shuffled)
            train_samples = shuffled[:3]
            test_samples = shuffled[3:]
            
        for s in train_samples:
            X_train_list.append(s["gabor_zm"])
            y_train_list.append(c_idx)
            filenames_train.append(s["filename"])
        for s in test_samples:
            X_test_list.append(s["gabor_zm"])
            y_test_list.append(c_idx)
            filenames_test.append(s["filename"])
            
    return (np.array(X_train_list), np.array(y_train_list), filenames_train,
            np.array(X_test_list), np.array(y_test_list), filenames_test)

# ---------------------------------------------------------
# Part 6: Paired Wilcoxon/t-test and Bootstrapping
# ---------------------------------------------------------
def paired_permutation_test(delta_eers, n_permutations=100000, seed=42):
    rng = np.random.default_rng(seed)
    obs_mean = np.mean(delta_eers)
    abs_obs_mean = np.abs(obs_mean)
    
    n_samples = len(delta_eers)
    # Vectorized permutation
    sign_flips = rng.choice([-1, 1], size=(n_permutations, n_samples))
    permuted_means = np.mean(sign_flips * delta_eers, axis=1)
    
    count = np.sum(np.abs(permuted_means) >= abs_obs_mean)
    p_val = count / n_permutations
    return p_val

def bootstrap_ci(delta_eers, n_bootstraps=2000, seed=42):
    rng = np.random.default_rng(seed)
    boot_means = []
    n_samples = len(delta_eers)
    for _ in range(n_bootstraps):
        boot_sample = rng.choice(delta_eers, size=n_samples, replace=True)
        boot_means.append(np.mean(boot_sample))
    boot_means = sorted(boot_means)
    low = boot_means[int(n_bootstraps * 0.025)]
    high = boot_means[int(n_bootstraps * 0.975)]
    return low, high

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def main():
    start_time = time.time()
    log_print("="*80)
    log_print("TECHNICAL RESOLUTION AND EXPERIMENTAL REPRODUCTION")
    log_print("="*80)
    
    # Load dataset
    loaded_data = load_all_iitd()
    n_classes = len(loaded_data)
    
    # ---------------------------------------------------------
    # PART A: Deterministic EER reproduction at k=128
    # ---------------------------------------------------------
    log_print("\n--- Running Deterministic Reproduction at k=128 ---")
    
    # 1. Standard Configuration
    std_kernels = get_gabor_kernels(ksize=15, sigma=4.0, lambd=8.0, gamma=0.5, orientations=None)
    std_loaded = {}
    for c, samples in loaded_data.items():
        std_loaded[c] = extract_gabor_for_samples(samples, std_kernels, (64, 64))
        
    X_tr_s, y_tr_s, fn_tr_s, X_te_s, y_te_s, fn_te_s = get_split_data(std_loaded, seed=None)
    
    pca_std = PCA(n_components=512, random_state=42)
    X_tr_s_proj_full = pca_std.fit_transform(X_tr_s)
    X_te_s_proj_full = pca_std.transform(X_te_s)
    
    k = 128
    X_tr_s_proj = X_tr_s_proj_full[:, :k]
    X_te_s_proj = X_te_s_proj_full[:, :k]
    
    # Baseline
    eer_std_base, auc_std_base, _, _, _ = evaluate_verification(X_tr_s_proj, y_tr_s, X_te_s_proj, y_te_s, n_classes)
    
    # XPCA (Residual, eta=0.2, weights=(0.2, 0.6, 0.2), shrinkage=0.1)
    # Let's write the config we used for the run
    with open("_paper_resolution/configs/standard_k128.json", "w") as f:
        f.write('{"k": 128, "orientations": 4, "lambda": 8.0, "sigma": 4.0, "gamma": 0.5, "downsample": [64, 64], "weights": [0.2, 0.6, 0.2], "shrinkage": 0.1, "eta": 0.2}')
        
    norm_util_s = get_xpca_weights(X_tr_s_proj, y_tr_s, pca_std.explained_variance_[:k], shrinkage=0.1, alpha=0.2, beta=0.6, gamma=0.2, bootstrap_iter=100, seed=42)
    weights_s = 1.0 + 0.2 * norm_util_s
    
    eer_std_xpca, auc_std_xpca, _, _, _ = evaluate_verification(X_tr_s_proj * weights_s, y_tr_s, X_te_s_proj * weights_s, y_te_s, n_classes)
    
    log_print(f"Standard Gabor+PCA Baseline EER: {eer_std_base*100:.6f}%")
    log_print(f"Standard Gabor+XPCA Residual EER: {eer_std_xpca*100:.6f}%")
    
    # 2. Optimized Configuration Candidate
    opt_kernels = get_gabor_kernels(ksize=15, sigma=5.0, lambd=10.0, gamma=0.5, orientations=[0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6])
    opt_loaded = {}
    for c, samples in loaded_data.items():
        opt_loaded[c] = extract_gabor_for_samples(samples, opt_kernels, (52, 52))
        
    X_tr_o, y_tr_o, fn_tr_o, X_te_o, y_te_o, fn_te_o = get_split_data(opt_loaded, seed=None)
    
    pca_opt = PCA(n_components=512, random_state=42)
    X_tr_o_proj_full = pca_opt.fit_transform(X_tr_o)
    X_te_o_proj_full = pca_opt.transform(X_te_o)
    
    X_tr_o_proj = X_tr_o_proj_full[:, :k]
    X_te_o_proj = X_te_o_proj_full[:, :k]
    
    # Baseline
    eer_opt_base, auc_opt_base, _, _, _ = evaluate_verification(X_tr_o_proj, y_tr_o, X_te_o_proj, y_te_o, n_classes)
    
    # XPCA (Optimized, Fisher Only (0, 1, 0), shrinkage=0.1, eta=0.02)
    with open("_paper_resolution/configs/optimized_k128.json", "w") as f:
        f.write('{"k": 128, "orientations": 6, "lambda": 10.0, "sigma": 5.0, "gamma": 0.5, "downsample": [52, 52], "weights": [0.0, 1.0, 0.0], "shrinkage": 0.1, "eta": 0.02}')
        
    norm_util_o = get_xpca_weights(X_tr_o_proj, y_tr_o, pca_opt.explained_variance_[:k], shrinkage=0.1, alpha=0.0, beta=1.0, gamma=0.0, bootstrap_iter=30, seed=42)
    weights_o = 1.0 + 0.02 * norm_util_o
    
    eer_opt_xpca, auc_opt_xpca, _, _, _ = evaluate_verification(X_tr_o_proj * weights_o, y_tr_o, X_te_o_proj * weights_o, y_te_o, n_classes)
    
    log_print(f"Optimized Gabor+PCA Baseline EER: {eer_opt_base*100:.6f}%")
    log_print(f"Optimized Gabor+XPCA (SOTA) EER: {eer_opt_xpca*100:.6f}%")
    
    # ---------------------------------------------------------
    # PART B: Paired Randomized Split Experiments (30 trials)
    # ---------------------------------------------------------
    log_print("\n--- Running 30 Paired Randomized Split Trials ---")
    
    trials_records = []
    
    # Define trials seeds
    trial_seeds = list(range(30))
    
    for t_id, seed in enumerate(trial_seeds):
        # 1. Standard Config Trial
        X_tr_s_t, y_tr_s_t, fn_tr_s_t, X_te_s_t, y_te_s_t, fn_te_s_t = get_split_data(std_loaded, seed=seed)
        
        # Save split filenames for reproducibility audit
        split_df = pd.DataFrame({"class": y_tr_s_t, "filename": fn_tr_s_t})
        split_df.to_csv(f"_paper_resolution/splits/trial_{t_id}_train_split.csv", index=False)
        
        pca_s_t = PCA(n_components=128, random_state=42)
        X_tr_s_t_proj = pca_s_t.fit_transform(X_tr_s_t)
        X_te_s_t_proj = pca_s_t.transform(X_te_s_t)
        
        # Baseline Standard
        eer_s_t_base, auc_s_t_base, th_s_t_base, gen_s_t_base, imp_s_t_base = evaluate_verification(
            X_tr_s_t_proj, y_tr_s_t, X_te_s_t_proj, y_te_s_t, n_classes
        )
        
        # XPCA Standard
        norm_util_s_t = get_xpca_weights(
            X_tr_s_t_proj, y_tr_s_t, pca_s_t.explained_variance_, shrinkage=0.1, alpha=0.2, beta=0.6, gamma=0.2, bootstrap_iter=100, seed=42
        )
        weights_s_t = 1.0 + 0.2 * norm_util_s_t
        
        eer_s_t_xpca, auc_s_t_xpca, th_s_t_xpca, gen_s_t_xpca, imp_s_t_xpca = evaluate_verification(
            X_tr_s_t_proj * weights_s_t, y_tr_s_t, X_te_s_t_proj * weights_s_t, y_te_s_t, n_classes
        )
        
        # Save raw scores for standard k=128 trial 0 as requested
        if t_id == 0:
            np.savez("_paper_resolution/raw_scores/trial_0_standard_scores.npz", 
                     gen_base=gen_s_t_base, imp_base=imp_s_t_base,
                     gen_xpca=gen_s_t_xpca, imp_xpca=imp_s_t_xpca)
            
        trials_records.append({
            "trial_id": t_id,
            "seed": seed,
            "split_id": f"split_{seed}",
            "k": 128,
            "config_name": "Standard",
            "baseline_eer": eer_s_t_base * 100,
            "xpca_eer": eer_s_t_xpca * 100,
            "delta_eer": (eer_s_t_xpca - eer_s_t_base) * 100,
            "baseline_auc": auc_s_t_base,
            "xpca_auc": auc_s_t_xpca,
            "delta_auc": auc_s_t_xpca - auc_s_t_base,
            "baseline_threshold": th_s_t_base,
            "xpca_threshold": th_s_t_xpca,
            "train_count": len(y_tr_s_t),
            "probe_count": len(y_te_s_t),
            "genuine_count": len(gen_s_t_base),
            "impostor_count": len(imp_s_t_base),
            "runtime": 0.0, # Filled later
            "git_commit": "N/A",
            "config_hash": "standard_k128",
            "score_file_baseline": f"trial_0_standard_scores.npz" if t_id == 0 else "",
            "score_file_xpca": f"trial_0_standard_scores.npz" if t_id == 0 else ""
        })
        
        # 2. Optimized Config Trial
        X_tr_o_t, y_tr_o_t, fn_tr_o_t, X_te_o_t, y_te_o_t, fn_te_o_t = get_split_data(opt_loaded, seed=seed)
        
        pca_o_t = PCA(n_components=128, random_state=42)
        X_tr_o_t_proj = pca_o_t.fit_transform(X_tr_o_t)
        X_te_o_t_proj = pca_o_t.transform(X_te_o_t)
        
        # Baseline Optimized
        eer_o_t_base, auc_o_t_base, th_o_t_base, gen_o_t_base, imp_o_t_base = evaluate_verification(
            X_tr_o_t_proj, y_tr_o_t, X_te_o_t_proj, y_te_o_t, n_classes
        )
        
        # XPCA Optimized
        norm_util_o_t = get_xpca_weights(
            X_tr_o_t_proj, y_tr_o_t, pca_o_t.explained_variance_, shrinkage=0.1, alpha=0.0, beta=1.0, gamma=0.0, bootstrap_iter=30, seed=42
        )
        weights_o_t = 1.0 + 0.02 * norm_util_o_t
        
        eer_o_t_xpca, auc_o_t_xpca, th_o_t_xpca, gen_o_t_xpca, imp_o_t_xpca = evaluate_verification(
            X_tr_o_t_proj * weights_o_t, y_tr_o_t, X_te_o_t_proj * weights_o_t, y_te_o_t, n_classes
        )
        
        if t_id == 0:
            np.savez("_paper_resolution/raw_scores/trial_0_optimized_scores.npz", 
                     gen_base=gen_o_t_base, imp_base=imp_o_t_base,
                     gen_xpca=gen_o_t_xpca, imp_xpca=imp_o_t_xpca)
            
        trials_records.append({
            "trial_id": t_id,
            "seed": seed,
            "split_id": f"split_{seed}",
            "k": 128,
            "config_name": "Optimized",
            "baseline_eer": eer_o_t_base * 100,
            "xpca_eer": eer_o_t_xpca * 100,
            "delta_eer": (eer_o_t_xpca - eer_o_t_base) * 100,
            "baseline_auc": auc_o_t_base,
            "xpca_auc": auc_o_t_xpca,
            "delta_auc": auc_o_t_xpca - auc_o_t_base,
            "baseline_threshold": th_o_t_base,
            "xpca_threshold": th_o_t_xpca,
            "train_count": len(y_tr_o_t),
            "probe_count": len(y_te_o_t),
            "genuine_count": len(gen_o_t_base),
            "impostor_count": len(imp_o_t_base),
            "runtime": 0.0,
            "git_commit": "N/A",
            "config_hash": "optimized_k128",
            "score_file_baseline": f"trial_0_optimized_scores.npz" if t_id == 0 else "",
            "score_file_xpca": f"trial_0_optimized_scores.npz" if t_id == 0 else ""
        })
        
    df_trials = pd.DataFrame(trials_records)
    # Fill in runtime roughly based on total elapsed
    elapsed = time.time() - start_time
    df_trials["runtime"] = elapsed / 60.0
    df_trials.to_csv("_paper_resolution/paired_trials.csv", index=False)
    log_print(f"Saved 60 trials records to _paper_resolution/paired_trials.csv")
    
    # ---------------------------------------------------------
    # PART C: Statistical Analyses
    # ---------------------------------------------------------
    log_print("\n--- Conducting Statistical Tests ---")
    
    stats_summary = []
    
    for cfg in ["Standard", "Optimized"]:
        df_cfg = df_trials[df_trials["config_name"] == cfg]
        
        base_eers = df_cfg["baseline_eer"].values
        xpca_eers = df_cfg["xpca_eer"].values
        delta_eers = df_cfg["delta_eer"].values
        
        base_aucs = df_cfg["baseline_auc"].values
        xpca_aucs = df_cfg["xpca_auc"].values
        delta_aucs = df_cfg["delta_auc"].values
        
        n_trials = len(df_cfg)
        
        # EER stats
        mean_base_eer = np.mean(base_eers)
        std_base_eer = np.std(base_eers, ddof=1)
        mean_xpca_eer = np.mean(xpca_eers)
        std_xpca_eer = np.std(xpca_eers, ddof=1)
        
        mean_delta_eer = np.mean(delta_eers)
        std_delta_eer = np.std(delta_eers, ddof=1)
        median_delta_eer = np.median(delta_eers)
        
        # Shapiro-Wilk for normality of delta_eer
        if np.all(delta_eers == 0):
            norm_stat, norm_p = 1.0, 1.0
        else:
            norm_stat, norm_p = stats.shapiro(delta_eers)
            
        # Paired tests
        t_stat, t_p = stats.ttest_rel(xpca_eers, base_eers)
        
        if np.all(delta_eers == 0):
            wilc_stat, wilc_p = 0.0, 1.0
        else:
            try:
                wilc_stat, wilc_p = stats.wilcoxon(xpca_eers, base_eers)
            except Exception as e:
                wilc_stat, wilc_p = 0.0, 1.0
                
        perm_p = paired_permutation_test(delta_eers)
        
        # CI for EERs and Delta EER
        ci_base_eer = stats.t.interval(0.95, df=n_trials-1, loc=mean_base_eer, scale=stats.sem(base_eers))
        ci_xpca_eer = stats.t.interval(0.95, df=n_trials-1, loc=mean_xpca_eer, scale=stats.sem(xpca_eers))
        ci_delta_eer = stats.t.interval(0.95, df=n_trials-1, loc=mean_delta_eer, scale=stats.sem(delta_eers))
        
        # CI bootstrap
        boot_low_eer, boot_high_eer = bootstrap_ci(delta_eers)
        
        # Wins/losses
        wins = np.sum(xpca_eers < base_eers)
        ties = np.sum(xpca_eers == base_eers)
        losses = np.sum(xpca_eers > base_eers)
        win_rate = wins / n_trials
        
        # Effect size (Cohen's d for paired)
        cohen_d = mean_delta_eer / (std_delta_eer + 1e-10)
        
        # AUC stats
        mean_base_auc = np.mean(base_aucs)
        std_base_auc = np.std(base_aucs, ddof=1)
        mean_xpca_auc = np.mean(xpca_aucs)
        std_xpca_auc = np.std(xpca_aucs, ddof=1)
        
        mean_delta_auc = np.mean(delta_aucs)
        std_delta_auc = np.std(delta_aucs, ddof=1)
        
        ci_base_auc = stats.t.interval(0.95, df=n_trials-1, loc=mean_base_auc, scale=stats.sem(base_aucs))
        ci_xpca_auc = stats.t.interval(0.95, df=n_trials-1, loc=mean_xpca_auc, scale=stats.sem(xpca_aucs))
        ci_delta_auc = stats.t.interval(0.95, df=n_trials-1, loc=mean_delta_auc, scale=stats.sem(delta_aucs))
        
        stats_summary.append({
            "config_name": cfg,
            "n_trials": n_trials,
            "mean_baseline_eer": mean_base_eer,
            "std_baseline_eer": std_base_eer,
            "ci95_baseline_eer_low": ci_base_eer[0],
            "ci95_baseline_eer_high": ci_base_eer[1],
            "mean_xpca_eer": mean_xpca_eer,
            "std_xpca_eer": std_xpca_eer,
            "ci95_xpca_eer_low": ci_xpca_eer[0],
            "ci95_xpca_eer_high": ci_xpca_eer[1],
            "mean_delta_eer": mean_delta_eer,
            "median_delta_eer": median_delta_eer,
            "std_delta_eer": std_delta_eer,
            "ci95_delta_eer_low": ci_delta_eer[0],
            "ci95_delta_eer_high": ci_delta_eer[1],
            "boot_ci95_delta_eer_low": boot_low_eer,
            "boot_ci95_delta_eer_high": boot_high_eer,
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "win_rate": win_rate,
            "cohen_d": cohen_d,
            "shapiro_stat": norm_stat,
            "shapiro_p": norm_p,
            "paired_t_stat": t_stat,
            "paired_t_p": t_p,
            "wilcoxon_stat": wilc_stat,
            "wilcoxon_p": wilc_p,
            "permutation_p": perm_p,
            "mean_baseline_auc": mean_base_auc,
            "std_baseline_auc": std_base_auc,
            "mean_xpca_auc": mean_xpca_auc,
            "std_xpca_auc": std_xpca_auc,
            "mean_delta_auc": mean_delta_auc,
            "std_delta_auc": std_delta_auc,
            "ci95_delta_auc_low": ci_delta_auc[0],
            "ci95_delta_auc_high": ci_delta_auc[1]
        })
        
        log_print(f"\n--- Results for {cfg} Configuration ---")
        log_print(f"Mean Baseline EER: {mean_base_eer:.4f}% | Mean XPCA EER: {mean_xpca_eer:.4f}%")
        log_print(f"Mean Delta EER: {mean_delta_eer:.4f}% (Median: {median_delta_eer:.4f}%)")
        log_print(f"95% CI of Delta: ({ci_delta_eer[0]:.4f}%, {ci_delta_eer[1]:.4f}%)")
        log_print(f"Bootstrap 95% CI of Delta: ({boot_low_eer:.4f}%, {boot_high_eer:.4f}%)")
        log_print(f"Wins: {wins} | Ties: {ties} | Losses: {losses} (Win Rate: {win_rate*100:.2f}%)")
        log_print(f"Cohen's d: {cohen_d:.4f}")
        log_print(f"Normality Shapiro-Wilk p: {norm_p:.6f}")
        log_print(f"Paired t-test p-value: {t_p:.6f}")
        log_print(f"Wilcoxon signed-rank p-value: {wilc_p:.6f}")
        log_print(f"Permutation test p-value: {perm_p:.6f}")
        log_print(f"Mean Baseline AUC: {mean_base_auc:.6f} | Mean XPCA AUC: {mean_xpca_auc:.6f}")
        log_print(f"Mean Delta AUC: {mean_delta_auc:.6f}")
        
    df_stats = pd.DataFrame(stats_summary)
    df_stats.to_csv("_paper_resolution/PAIRED_STATISTICAL_SUMMARY.csv", index=False)
    log_print(f"\nSaved statistical summary to _paper_resolution/PAIRED_STATISTICAL_SUMMARY.csv")
    log_file.close()

if __name__ == "__main__":
    main()
