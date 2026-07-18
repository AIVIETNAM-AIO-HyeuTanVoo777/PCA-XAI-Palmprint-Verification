import os
import sys
import glob
import cv2
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.stats import ttest_rel
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from skimage.metrics import structural_similarity as ssim_func

# Add src directory to python path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from evaluation.metrics import evaluate_verification

def compute_psnr(original, reconstructed):
    """Computes Peak Signal-to-Noise Ratio (PSNR) for normalized images [0, 1]."""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

def load_dataset_metadata(data_dir="data/IITD", target_size=(128, 128)):
    """
    Loads all images from the IITD Segmented directory and extracts metadata.
    Ensures exact class indexing and ordering.
    """
    left_dir = os.path.join(data_dir, "Segmented", "Left")
    right_dir = os.path.join(data_dir, "Segmented", "Right")
    
    # Fallback to local IITD Dataset directory
    if not os.path.exists(left_dir) or not os.path.exists(right_dir):
        left_dir = "IITD Dataset/Segmented/Left"
        right_dir = "IITD Dataset/Segmented/Right"
        if not os.path.exists(left_dir) or not os.path.exists(right_dir):
            raise FileNotFoundError("IITD Segmented directory not found.")
        
    left_files = glob.glob(os.path.join(left_dir, "*.bmp"))
    right_files = glob.glob(os.path.join(right_dir, "*.bmp"))
    
    all_files = []
    for f in left_files:
        all_files.append((f, 'L'))
    for f in right_files:
        all_files.append((f, 'R'))
        
    palm_groups = {}
    for f, side in all_files:
        basename = os.path.basename(f)
        parts = basename.split("_")
        if len(parts) < 2:
            continue
        subj_id = int(parts[0])
        palm_key = f"{subj_id}_{side}"
        if palm_key not in palm_groups:
            palm_groups[palm_key] = []
        palm_groups[palm_key].append(f)
        
    # Sort palm keys to match loader.py
    sorted_keys = sorted(palm_groups.keys(), key=lambda x: (int(x.split("_")[0]), x.split("_")[1]))
    class_map = {k: i for i, k in enumerate(sorted_keys)}
    
    records = []
    for palm_key in sorted_keys:
        class_idx = class_map[palm_key]
        subj_id = int(palm_key.split("_")[0])
        side = palm_key.split("_")[1]
        
        files = palm_groups[palm_key]
        
        # Sort files by sample number
        def get_sample_id(f):
            basename = os.path.basename(f)
            parts = basename.split("_")
            try:
                return int(parts[1].split(".")[0])
            except ValueError:
                return 999
        files_sorted = sorted(files, key=get_sample_id)
        
        for f in files_sorted:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            img_zm = img_normalized - np.mean(img_normalized)
            
            records.append({
                "img_zm": img_zm.flatten(),
                "img_raw": img_normalized.flatten(),
                "subject_id": subj_id,
                "palm_class": class_idx,
                "side": side,
                "sample_id": get_sample_id(f),
                "file_path": f
            })
            
    return records

def main():
    print("=" * 70)
    print("RUNNING ACADEMIC EVALUATION ENGINE FOR EXPLAINABLE PCA (STRESS TEST)")
    print("=" * 70)

    # Create output directories
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    # Load dataset with full metadata
    records = load_dataset_metadata()
    print(f"Loaded {len(records)} images with metadata.")

    # Reconstruct train/test matrices for baseline closed-set evaluation
    # Baseline partition: first 3 images per palm class -> train, rest -> test
    X_train, y_train = [], []
    X_test, y_test = [], []
    test_means = []

    # Group records by palm class to perform splitting
    palm_class_records = {}
    for r in records:
        c = r["palm_class"]
        if c not in palm_class_records:
            palm_class_records[c] = []
        palm_class_records[c].append(r)

    for c in sorted(palm_class_records.keys()):
        class_recs = palm_class_records[c]
        # First 3 go to train, remaining to test
        train_recs = class_recs[:3]
        test_recs = class_recs[3:]
        
        for r in train_recs:
            X_train.append(r["img_zm"])
            y_train.append(r["palm_class"])
        for r in test_recs:
            X_test.append(r["img_zm"])
            y_test.append(r["palm_class"])
            test_means.append(np.mean(r["img_raw"]))

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    test_means = np.array(test_means)
    n_classes = len(palm_class_records)
    D = X_train.shape[1] # 16384

    # Fit PCA on training set (512 components)
    print("\nFitting SVD-based PCA model (512 components)...")
    pca = PCA(n_components=512, random_state=42)
    pca.fit(X_train)
    eigvals = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Retrieve global training mean vector
    global_mean = pca.mean_

    # Project training and test sets
    train_proj = pca.transform(X_train)
    test_proj = pca.transform(X_test)

    # --- EXPERIMENT 1: DATASET-WIDE RECONSTRUCTION ANALYSIS ---
    print("\nRunning Experiment 1: Dataset-wide Reconstruction Analysis...")
    dimensions = [16, 32, 64, 128, 256, 512]
    
    test_reconstruction_metrics = {k: [] for k in dimensions}

    for k in dimensions:
        components = pca.components_[:k]
        for idx in range(len(X_test)):
            proj = test_proj[idx, :k]
            sample = X_test[idx]
            
            # Reconstruct (zero-mean space)
            recon = proj @ components + global_mean
            
            # Denormalize by adding back the individual sample mean
            orig_denorm = np.clip(sample + test_means[idx], 0.0, 1.0)
            recon_denorm = np.clip(recon + test_means[idx], 0.0, 1.0)
            
            orig_img = orig_denorm.reshape((128, 128))
            recon_img = recon_denorm.reshape((128, 128))
            
            mse = np.mean((orig_denorm - recon_denorm) ** 2)
            psnr = compute_psnr(orig_denorm, recon_denorm)
            ssim = ssim_func(orig_img, recon_img, data_range=1.0)
            
            test_reconstruction_metrics[k].append({
                "index": idx,
                "mse": mse,
                "psnr": psnr,
                "ssim": ssim
            })

    # Summarize dataset-wide reconstruction
    recon_summary = []
    for k in dimensions:
        mses = [m["mse"] for m in test_reconstruction_metrics[k]]
        psnrs = [m["psnr"] for m in test_reconstruction_metrics[k]]
        ssims = [m["ssim"] for m in test_reconstruction_metrics[k]]
        
        recon_summary.append({
            "k": k,
            "mse_mean": np.mean(mses),
            "mse_std": np.std(mses),
            "psnr_mean": np.mean(psnrs),
            "psnr_std": np.std(psnrs),
            "ssim_mean": np.mean(ssims),
            "ssim_std": np.std(ssims)
        })
    df_recon = pd.DataFrame(recon_summary)
    df_recon.to_csv("results/tables/reconstruction_dataset_wide.csv", index=False)
    print("Saved results/tables/reconstruction_dataset_wide.csv")

    # SSIM ranking at k=128 to identify Best, Median, Worst
    ssims_128 = test_reconstruction_metrics[128]
    ssims_128_sorted = sorted(ssims_128, key=lambda x: x["ssim"])
    worst_recon = ssims_128_sorted[0]
    best_recon = ssims_128_sorted[-1]
    median_recon = ssims_128_sorted[len(ssims_128_sorted) // 2]
    
    # Save ranking sample details
    df_samples = pd.DataFrame([
        {"Category": "Worst", "Index": worst_recon["index"], "SSIM": worst_recon["ssim"], "PSNR": worst_recon["psnr"], "MSE": worst_recon["mse"]},
        {"Category": "Median", "Index": median_recon["index"], "SSIM": median_recon["ssim"], "PSNR": median_recon["psnr"], "MSE": median_recon["mse"]},
        {"Category": "Best", "Index": best_recon["index"], "SSIM": best_recon["ssim"], "PSNR": best_recon["psnr"], "MSE": best_recon["mse"]}
    ])
    df_samples.to_csv("results/tables/reconstruction_representative_samples.csv", index=False)

    # --- EXPERIMENT 2: STABILIZED FISHER COMPONENT IMPORTANCE ---
    print("\nRunning Experiment 2: Stabilized Fisher Component Importance...")
    fisher_scores = []
    eps = 1e-6
    shrinkage = 0.1 # regularize class variance with global variance

    for i in range(512):
        z = train_proj[:, i]
        class_means = []
        class_vars = []
        for c in range(n_classes):
            mask = (y_train == c)
            z_c = z[mask]
            if len(z_c) > 0:
                class_means.append(np.mean(z_c))
                class_vars.append(np.var(z_c))
            else:
                class_means.append(0.0)
                class_vars.append(0.0)
        
        var_inter = np.var(class_means)
        var_intra = np.mean(class_vars)
        var_global = eigvals[i]
        
        var_intra_reg = (1 - shrinkage) * var_intra + shrinkage * var_global
        f_score = var_inter / (var_intra_reg + eps)
        fisher_scores.append(f_score)

    fisher_scores = np.array(fisher_scores)
    
    fisher_ranking = np.argsort(fisher_scores)[::-1] # descending
    df_fisher = pd.DataFrame({
        "PC_Index": np.arange(1, 513),
        "Explained_Variance": eigvals,
        "Variance_Ratio": explained_variance_ratio,
        "Fisher_Score": fisher_scores,
        "Fisher_Rank": np.argsort(np.argsort(fisher_scores)[::-1]) + 1
    })
    df_fisher.to_csv("results/tables/fisher_scores_ranking.csv", index=False)
    print("Saved results/tables/fisher_scores_ranking.csv")

    # Quantify correlation between explained variance and Fisher Score
    pearson_corr = np.corrcoef(eigvals, fisher_scores)[0, 1]
    print(f"Pearson correlation between PCA explained variance and Fisher Score: {pearson_corr:.4f}")

    # --- EXPERIMENT 3: COMPONENT ABLATION STUDY ---
    print("\nRunning Experiment 3: PCA Component Ablation Study...")
    ablation_results = []

    # Helper function to evaluate verification for a specific subset of component indices
    def evaluate_subset(pc_indices):
        tr = train_proj[:, pc_indices]
        te = test_proj[:, pc_indices]
        res = evaluate_verification(tr, y_train, te, y_test, n_classes)
        return res["accuracy"], res["eer"], res["auc"]

    # Evaluates A-F: PC1-k
    for k in dimensions:
        acc, eer, auc = evaluate_subset(list(range(k)))
        ablation_results.append({
            "Experiment": f"PC1-{k}",
            "k": k,
            "Accuracy": acc,
            "EER": eer,
            "AUC": auc
        })

    # Evaluates G: Only Top-k Fisher-ranked components
    for k in dimensions:
        top_k_fisher_indices = fisher_ranking[:k]
        acc, eer, auc = evaluate_subset(top_k_fisher_indices)
        ablation_results.append({
            "Experiment": f"Top-{k} Fisher PCs",
            "k": k,
            "Accuracy": acc,
            "EER": eer,
            "AUC": auc
        })

    # Evaluates H: Remove Top-5 Fisher PCs from PC1-128
    base_128_indices = list(range(128))
    top_5_fisher_global_indices = [idx for idx in fisher_ranking[:5] if idx < 128]
    h_indices = [idx for idx in base_128_indices if idx not in top_5_fisher_global_indices]
    acc_h, eer_h, auc_h = evaluate_subset(h_indices)
    ablation_results.append({
        "Experiment": "PC1-128 (Remove Top-5 Fisher)",
        "k": len(h_indices),
        "Accuracy": acc_h,
        "EER": eer_h,
        "AUC": auc_h
    })

    # Evaluates I: Remove Bottom-50 Fisher PCs from PC1-128
    bottom_50_fisher_global_indices = [idx for idx in fisher_ranking[-50:] if idx < 128]
    i_indices = [idx for idx in base_128_indices if idx not in bottom_50_fisher_global_indices]
    acc_i, eer_i, auc_i = evaluate_subset(i_indices)
    ablation_results.append({
        "Experiment": "PC1-128 (Remove Bottom-50 Fisher)",
        "k": len(i_indices),
        "Accuracy": acc_i,
        "EER": eer_i,
        "AUC": auc_i
    })

    df_ablation = pd.DataFrame(ablation_results)
    df_ablation.to_csv("results/tables/component_ablation.csv", index=False)
    print("Saved results/tables/component_ablation.csv")

    # --- EXPERIMENT 4: BASELINE STRENGTHENING (PCA+LDA) ---
    print("\nRunning Experiment 4: Fitting PCA+LDA Supervised Baseline...")
    X_tr_pca128 = train_proj[:, :128]
    X_te_pca128 = test_proj[:, :128]

    lda = LDA()
    lda.fit(X_tr_pca128, y_train)

    train_proj_lda = lda.transform(X_tr_pca128)
    test_proj_lda = lda.transform(X_te_pca128)

    res_lda = evaluate_verification(train_proj_lda, y_train, test_proj_lda, y_test, n_classes)
    
    # Save baseline comparison
    df_lda_comp = pd.DataFrame([{
        "Method": "PCA (Unsupervised)",
        "k": 128,
        "Accuracy": df_ablation[df_ablation["Experiment"] == "PC1-128"]["Accuracy"].values[0],
        "EER": df_ablation[df_ablation["Experiment"] == "PC1-128"]["EER"].values[0],
        "AUC": df_ablation[df_ablation["Experiment"] == "PC1-128"]["AUC"].values[0]
    }, {
        "Method": "PCA+LDA (Supervised)",
        "k": train_proj_lda.shape[1],
        "Accuracy": res_lda["accuracy"],
        "EER": res_lda["eer"],
        "AUC": res_lda["auc"]
    }])
    df_lda_comp.to_csv("results/tables/baseline_comparison.csv", index=False)
    print("Saved results/tables/baseline_comparison.csv")

    # --- EXPERIMENT 5: CLOSED-SET STATISTICAL SATURATION ANALYSIS (20 runs) ---
    print("\nRunning Experiment 5: Running closed-set statistical validation over 20 randomized splits...")
    seeds = list(range(42, 62)) # 20 seeds
    run_eers = {k: [] for k in dimensions}
    run_accs = {k: [] for k in dimensions}
    run_aucs = {k: [] for k in dimensions}

    # Gather subjects map
    subject_to_records = {}
    for r in records:
        s = r["subject_id"]
        if s not in subject_to_records:
            subject_to_records[s] = []
        subject_to_records[s].append(r)

    # In each run we randomly split the images within each class (closed-set)
    for run_idx, seed in enumerate(seeds):
        X_tr_r, y_tr_r = [], []
        X_te_r, y_te_r = [], []
        
        # Shuffle images for each class using class-specific rng
        for c in sorted(palm_class_records.keys()):
            class_recs = palm_class_records[c]
            rng = np.random.default_rng(seed + c)
            shuffled_recs = list(class_recs)
            rng.shuffle(shuffled_recs)
            
            train_recs = shuffled_recs[:3]
            test_recs = shuffled_recs[3:]
            
            for r in train_recs:
                X_tr_r.append(r["img_zm"])
                y_tr_r.append(r["palm_class"])
            for r in test_recs:
                X_te_r.append(r["img_zm"])
                y_te_r.append(r["palm_class"])
                
        X_tr_r = np.array(X_tr_r)
        y_tr_r = np.array(y_tr_r)
        X_te_r = np.array(X_te_r)
        y_te_r = np.array(y_te_r)
        
        pca_r = PCA(n_components=512, random_state=42)
        pca_r.fit(X_tr_r)
        
        tr_proj_r = pca_r.transform(X_tr_r)
        te_proj_r = pca_r.transform(X_te_r)
        
        for k in dimensions:
            res_r = evaluate_verification(tr_proj_r[:, :k], y_tr_r, te_proj_r[:, :k], y_te_r, n_classes)
            run_eers[k].append(res_r["eer"])
            run_accs[k].append(res_r["accuracy"])
            run_aucs[k].append(res_r["auc"])

    # Compute statistics summary (Mean, Std, 95% Confidence Intervals using t-distribution)
    t_critical = st.t.ppf(0.975, df=19) # ~2.093
    
    summary_stats = []
    for k in dimensions:
        mean_eer = np.mean(run_eers[k])
        std_eer = np.std(run_eers[k], ddof=1)
        sem_eer = std_eer / np.sqrt(20)
        ci_eer = t_critical * sem_eer
        
        summary_stats.append({
            "k": k,
            "eer_mean": mean_eer,
            "eer_std": std_eer,
            "eer_ci_half": ci_eer,
            "accuracy_mean": np.mean(run_accs[k]),
            "accuracy_std": np.std(run_accs[k], ddof=1),
            "auc_mean": np.mean(run_aucs[k]),
            "auc_std": np.std(run_aucs[k], ddof=1)
        })
    df_stats = pd.DataFrame(summary_stats)
    df_stats.to_csv("results/tables/statistical_summary.csv", index=False)
    print("Saved results/tables/statistical_summary.csv")

    # Run paired t-tests and Cohen's d effect sizes
    comparisons = [
        (16, 32), (32, 64), (64, 128), (128, 256), (256, 512),
        (16, 512), (32, 512), (64, 512), (128, 512), (256, 512)
    ]
    ttest_results = []
    bonferroni_thresh = 0.05 / len(comparisons) # 0.005 significance threshold
    
    for d1, d2 in comparisons:
        eers1 = run_eers[d1]
        eers2 = run_eers[d2]
        t_stat, p_val = ttest_rel(eers1, eers2)
        diff = np.array(eers1) - np.array(eers2)
        cohens_d = np.mean(diff) / np.std(diff, ddof=1)
        sig = "Yes" if p_val < bonferroni_thresh else "No"
        
        ttest_results.append({
            "Comparison": f"{d1} vs {d2}",
            "t_statistic": t_stat,
            "p_value": p_val,
            "Bonferroni_Threshold": bonferroni_thresh,
            "Statistically_Significant": sig,
            "Cohens_d": cohens_d
        })
    df_ttest = pd.DataFrame(ttest_results)
    df_ttest.to_csv("results/tables/statistical_significance.csv", index=False)
    print("Saved results/tables/statistical_significance.csv")

    # --- EXPERIMENT 5B: OPEN-SET (SUBJECT-DISJOINT) GENERALIZATION STRESS TEST ---
    print("\nRunning Experiment 5B: Running subject-disjoint open-set validation over 20 randomized splits...")
    open_run_eers = {k: [] for k in dimensions}
    open_run_accs = {k: [] for k in dimensions}
    open_run_aucs = {k: [] for k in dimensions}
    
    # We select subjects from the keys of subject_to_records
    unique_subject_ids = list(subject_to_records.keys())
    num_subjects = len(unique_subject_ids) # should be 230
    
    for run_idx, seed in enumerate(seeds):
        # Deterministic subject split per run
        rng = np.random.default_rng(seed)
        shuffled_subjects = list(unique_subject_ids)
        rng.shuffle(shuffled_subjects)
        
        train_subjects = shuffled_subjects[:150] # 150 subjects for PCA fit
        test_subjects = shuffled_subjects[150:]  # 80 subjects for open-set verification
        
        # Build training set images
        X_tr_r = []
        for s in train_subjects:
            for r in subject_to_records[s]:
                X_tr_r.append(r["img_zm"])
        X_tr_r = np.array(X_tr_r)
        
        # Fit PCA on training subject images (completely disjoint from test subjects)
        pca_r = PCA(n_components=512, random_state=42)
        pca_r.fit(X_tr_r)
        
        # On the 80 test subjects, we need to map their palm classes to new consecutive labels
        test_X_train = []
        test_y_train = []
        test_X_test = []
        test_y_test = []
        
        new_class_idx = 0
        for s in test_subjects:
            # Sort records of this subject by side ('L' or 'R')
            left_recs = [r for r in subject_to_records[s] if r["side"] == 'L']
            right_recs = [r for r in subject_to_records[s] if r["side"] == 'R']
            
            # Left side split
            if len(left_recs) >= 3:
                left_recs_sorted = sorted(left_recs, key=lambda x: x["sample_id"])
                train_recs = left_recs_sorted[:3]
                test_recs = left_recs_sorted[3:]
                
                for r in train_recs:
                    test_X_train.append(r["img_zm"])
                    test_y_train.append(new_class_idx)
                for r in test_recs:
                    test_X_test.append(r["img_zm"])
                    test_y_test.append(new_class_idx)
                new_class_idx += 1
                
            # Right side split
            if len(right_recs) >= 3:
                right_recs_sorted = sorted(right_recs, key=lambda x: x["sample_id"])
                train_recs = right_recs_sorted[:3]
                test_recs = right_recs_sorted[3:]
                
                for r in train_recs:
                    test_X_train.append(r["img_zm"])
                    test_y_train.append(new_class_idx)
                for r in test_recs:
                    test_X_test.append(r["img_zm"])
                    test_y_test.append(new_class_idx)
                new_class_idx += 1
                
        test_X_train = np.array(test_X_train)
        test_y_train = np.array(test_y_train)
        test_X_test = np.array(test_X_test)
        test_y_test = np.array(test_y_test)
        n_classes_test = new_class_idx
        
        # Project test subjects' images using PCA fitted on training subjects
        tr_proj_r = pca_r.transform(test_X_train)
        te_proj_r = pca_r.transform(test_X_test)
        
        for k in dimensions:
            res_r = evaluate_verification(tr_proj_r[:, :k], test_y_train, te_proj_r[:, :k], test_y_test, n_classes_test)
            open_run_eers[k].append(res_r["eer"])
            open_run_accs[k].append(res_r["accuracy"])
            open_run_aucs[k].append(res_r["auc"])

    # Compute open-set statistics summary
    open_summary_stats = []
    for k in dimensions:
        mean_eer = np.mean(open_run_eers[k])
        std_eer = np.std(open_run_eers[k], ddof=1)
        sem_eer = std_eer / np.sqrt(20)
        ci_eer = t_critical * sem_eer
        
        open_summary_stats.append({
            "k": k,
            "eer_mean": mean_eer,
            "eer_std": std_eer,
            "eer_ci_half": ci_eer,
            "accuracy_mean": np.mean(open_run_accs[k]),
            "accuracy_std": np.std(open_run_accs[k], ddof=1),
            "auc_mean": np.mean(open_run_aucs[k]),
            "auc_std": np.std(open_run_aucs[k], ddof=1)
        })
    df_open_stats = pd.DataFrame(open_summary_stats)
    df_open_stats.to_csv("results/tables/open_set_statistical_summary.csv", index=False)
    print("Saved results/tables/open_set_statistical_summary.csv")

    # Run paired t-tests for open-set split
    open_ttest_results = []
    for d1, d2 in comparisons:
        eers1 = open_run_eers[d1]
        eers2 = open_run_eers[d2]
        t_stat, p_val = ttest_rel(eers1, eers2)
        diff = np.array(eers1) - np.array(eers2)
        cohens_d = np.mean(diff) / np.std(diff, ddof=1)
        sig = "Yes" if p_val < bonferroni_thresh else "No"
        
        open_ttest_results.append({
            "Comparison": f"{d1} vs {d2}",
            "t_statistic": t_stat,
            "p_value": p_val,
            "Bonferroni_Threshold": bonferroni_thresh,
            "Statistically_Significant": sig,
            "Cohens_d": cohens_d
        })
    df_open_ttest = pd.DataFrame(open_ttest_results)
    df_open_ttest.to_csv("results/tables/open_set_statistical_significance.csv", index=False)
    print("Saved results/tables/open_set_statistical_significance.csv")

    # --- EXPERIMENT 6: CAES++ (IB FORMULATION) ---
    print("\nRunning Experiment 6: Redesigning and evaluating CAES++...")
    mse_vals = df_recon["mse_mean"].values
    
    caes_records = []
    beta_base, gamma_base, mu_base = 1.0, 1.0, 0.05
    sigma2 = 1.0

    for idx, k in enumerate(dimensions):
        # Rate: Shannon entropy / Gaussian mutual info surrogate
        rate = np.sum(np.log(1 + eigvals[:k] / sigma2))
        
        # Distortion: Sum of squared errors
        distortion = D * mse_vals[idx]
        
        # Discriminative information: log-Fisher gain surrogate (Mutual Info proxy)
        discrim = np.sum(np.log(1 + fisher_scores[:k]))
        
        # Complexity penalty: MDL complexity penalty
        complexity = k * np.log(D)
        
        # CAES++ score
        caes_val = rate - beta_base * distortion + gamma_base * discrim - mu_base * complexity
        
        caes_records.append({
            "k": k,
            "Rate": rate,
            "Distortion": distortion,
            "Discriminative": discrim,
            "Complexity": complexity,
            "CAES_plus": caes_val
        })
    df_caes = pd.DataFrame(caes_records)
    df_caes.to_csv("results/tables/caes_scores.csv", index=False)
    print("Saved results/tables/caes_scores.csv")

    # Global Sensitivity Grid-search Analysis
    beta_grid = np.linspace(0.1, 5.0, 10)
    gamma_grid = np.linspace(0.1, 5.0, 10)
    mu_grid = np.linspace(0.01, 0.2, 10)
    
    grid_results = []
    for b in beta_grid:
        for g in gamma_grid:
            for m in mu_grid:
                scores = []
                for idx, k in enumerate(dimensions):
                    rate = caes_records[idx]["Rate"]
                    dist = caes_records[idx]["Distortion"]
                    disc = caes_records[idx]["Discriminative"]
                    comp = caes_records[idx]["Complexity"]
                    score = rate - b * dist + g * disc - m * comp
                    scores.append(score)
                optimal_k = dimensions[np.argmax(scores)]
                grid_results.append(optimal_k)
                
    grid_results = np.array(grid_results)
    unique_ks, counts = np.unique(grid_results, return_counts=True)
    
    df_sensitivity = pd.DataFrame({
        "Optimal_k": unique_ks,
        "Grid_Counts": counts,
        "Percentage": (counts / len(grid_results)) * 100.0
    })
    df_sensitivity.to_csv("results/tables/caes_sensitivity.csv", index=False)

    # --- EXPERIMENT 7: EXPLORATORY CORRELATION ANALYSIS ---
    print("\nRunning Experiment 7: Computing metric correlation matrix...")
    allowed_experiments = [f"PC1-{k}" for k in dimensions]
    acc_vals = df_ablation[df_ablation["Experiment"].isin(allowed_experiments)]["Accuracy"].values
    eer_vals = df_ablation[df_ablation["Experiment"].isin(allowed_experiments)]["EER"].values
    var_vals = cumulative_variance[[k - 1 for k in dimensions]]
    
    df_corr_data = pd.DataFrame({
        "Variance_Retention": var_vals,
        "SSIM": df_recon["ssim_mean"].values,
        "PSNR": df_recon["psnr_mean"].values,
        "EER": eer_vals,
        "Accuracy": acc_vals
    })
    
    corr_matrix = df_corr_data.corr(method="pearson")
    corr_matrix.to_csv("results/tables/correlation_matrix.csv")
    print("Saved results/tables/correlation_matrix.csv")

    # --- EXPERIMENT 8: COMPRESSION TRADE-OFF ---
    print("\nRunning Experiment 8: Building template compression trade-off table...")
    raw_size_bytes = D * 4
    compression_tradeoffs = []
    for idx, k in enumerate(dimensions):
        template_size_bytes = k * 4
        comp_ratio = raw_size_bytes / template_size_bytes
        compression_tradeoffs.append({
            "k": k,
            "Variance_Retention_Percent": var_vals[idx] * 100.0,
            "Accuracy_Percent": acc_vals[idx] * 100.0,
            "EER_Percent": eer_vals[idx] * 100.0,
            "Template_Size_Bytes": template_size_bytes,
            "Compression_Ratio": comp_ratio
        })
    df_comp_ratio = pd.DataFrame(compression_tradeoffs)
    df_comp_ratio.to_csv("results/tables/compression_tradeoff.csv", index=False)
    print("Saved results/tables/compression_tradeoff.csv")

    print("\n" + "=" * 70)
    print("SUCCESS: All academic evaluations and CSV tables generated!")
    print("=" * 70)

if __name__ == "__main__":
    main()
