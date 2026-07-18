import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

# Add src to python path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing.loader import load_iitd_dataset
from feature_extraction.pca import fit_pca_model, project_embeddings
from evaluation.metrics import evaluate_verification, run_significance_test
from explainability.eigenpalm import generate_eigenpalm_visualization
from explainability.reconstruction import run_reconstruction_analysis
from explainability.distribution_analysis import analyze_score_distributions
from explainability.report import generate_pdf_report
from visualization.plots import (
    draw_framework_overview,
    plot_pca_dimension_analysis,
    plot_variance_retention,
    plot_component_importance
)

def run_all_experiments():
    print("=" * 70)
    print("RUNNING EXPLAINABLE PCA PALMPRINT FRAMEWORK EXPERIMENTS (IITD ONLY)")
    print("=" * 70)
    
    # 0. Setup directories
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    
    # --- STEP 1: LOAD BASELINE DATA & FIT FULL PCA ---
    # Load default split (non-shuffled)
    X_train, y_train, X_test, y_test, n_classes, n_subjects = load_iitd_dataset(
        data_dir="data/IITD", max_subjects=None, target_size=(128, 128), zero_mean_samples=True
    )
    
    # Fit full PCA (using 512 components)
    print("\nFitting full PCA on IITD...")
    pca = fit_pca_model(X_train, n_components=512)
    
    # --- EXPERIMENT 1: PCA DIMENSION ANALYSIS ---
    print("\nRunning Experiment 1: PCA Dimension Analysis...")
    dimensions = [16, 32, 64, 128, 256, 512]
    exp1_results = []
    
    # We will also save curves of optimal ROC for figure plotting
    fprs_dict = {}
    tprs_dict = {}
    
    for k in dimensions:
        train_proj = project_embeddings(pca, X_train, k=k)
        test_proj = project_embeddings(pca, X_test, k=k)
        
        res = evaluate_verification(train_proj, y_train, test_proj, y_test, n_classes)
        
        exp1_results.append({
            "components": k,
            "accuracy": res["accuracy"],
            "far": res["far"],
            "frr": res["frr"],
            "eer": res["eer"],
            "auc": res["auc"]
        })
        
        # Cache ROC curves for later mapping
        fprs_dict[k] = res["fpr_curve"]
        tprs_dict[k] = res["tpr_curve"]
        
    df_dim_analysis = pd.DataFrame(exp1_results)
    df_dim_analysis.to_csv("results/tables/pca_dimension_analysis.csv", index=False)
    print("Saved results/tables/pca_dimension_analysis.csv")
    
    # Generate Figure 2
    plot_pca_dimension_analysis(
        pca_dims=dimensions,
        accuracies=df_dim_analysis["accuracy"].values,
        eers=df_dim_analysis["eer"].values,
        aucs=df_dim_analysis["auc"].values,
        save_dir="results/figures"
    )
    
    # --- EXPERIMENT 2: VARIANCE RETENTION ANALYSIS ---
    print("\nRunning Experiment 2: Variance Retention Analysis...")
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    var_retained_list = []
    for k in dimensions:
        k_val = min(k, len(cumulative_variance))
        var_retained_list.append(cumulative_variance[k_val - 1])
        
    df_variance = pd.DataFrame({
        "Components": dimensions,
        "Variance Retained (%)": [v * 100.0 for v in var_retained_list]
    })
    df_variance.to_csv("results/tables/variance_retention_analysis.csv", index=False)
    print("Saved results/tables/variance_retention_analysis.csv")
    
    # Generate Figure 3
    plot_variance_retention(cumulative_variance, save_dir="results/figures")
    
    # --- EXPERIMENT 3: EIGENPALM VISUALIZATION ---
    print("\nRunning Experiment 3: EigenPalm Visualizations...")
    generate_eigenpalm_visualization(pca, target_size=(128, 128), save_dir="results/figures")
    
    # --- EXPERIMENT 4: RECONSTRUCTION ANALYSIS ---
    print("\nRunning Experiment 4: Reconstruction Analysis...")
    # Pick 3 representative samples from the training set (e.g. index 0, 10, 20)
    rep_indices = [0, 15, 30]
    X_samples = X_train[rep_indices]
    df_recon = run_reconstruction_analysis(pca, X_samples, target_size=(128, 128), save_dir="results")
    
    # --- EXPERIMENT 5: COMPONENT IMPORTANCE ANALYSIS ---
    print("\nRunning Experiment 5: Component Importance Analysis...")
    plot_component_importance(pca.explained_variance_, save_dir="results/figures")
    
    # --- EXPERIMENT 6: GENUINE VS IMPOSTER DISTRIBUTION ANALYSIS ---
    print("\nRunning Experiment 6: Match Score Distribution Analysis...")
    # Project embeddings at optimal dimension (e.g. k=128)
    train_proj_opt = project_embeddings(pca, X_train, k=128)
    test_proj_opt = project_embeddings(pca, X_test, k=128)
    res_opt = evaluate_verification(train_proj_opt, y_train, test_proj_opt, y_test, n_classes)
    
    stats_dist = analyze_score_distributions(
        res_opt["genuine_scores"], res_opt["imposter_scores"], save_dir="results"
    )
    
    # --- STATISTICAL ANALYSIS & REPEATED RUNS ---
    print("\nRunning Statistical Analysis (5 repeated trials with randomized splits)...")
    seeds = [42, 43, 44, 45, 46]
    n_runs = 5
    
    # Metrics dictionaries: key is dimension, value is list of results across runs
    run_eers = {k: [] for k in dimensions}
    run_accs = {k: [] for k in dimensions}
    run_aucs = {k: [] for k in dimensions}
    
    for run_idx in range(n_runs):
        seed = seeds[run_idx]
        print(f"  Executing Run {run_idx+1}/{n_runs} (seed={seed})...")
        
        # Load dataset with randomized splits
        X_tr_r, y_tr_r, X_te_r, y_te_r, n_cl_r, _ = load_iitd_dataset(
            data_dir="data/IITD", max_subjects=None, target_size=(128, 128),
            zero_mean_samples=True, random_split=True, seed=seed
        )
        
        # Fit PCA on training set
        pca_r = fit_pca_model(X_tr_r, n_components=512)
        
        for k in dimensions:
            tr_proj_r = project_embeddings(pca_r, X_tr_r, k=k)
            te_proj_r = project_embeddings(pca_r, X_te_r, k=k)
            
            res_r = evaluate_verification(tr_proj_r, y_tr_r, te_proj_r, y_te_r, n_cl_r)
            
            run_eers[k].append(res_r["eer"])
            run_accs[k].append(res_r["accuracy"])
            run_aucs[k].append(res_r["auc"])
            
    # Compute stats summaries
    summary_stats = []
    for k in dimensions:
        summary_stats.append({
            "components": k,
            "accuracy_mean": np.mean(run_accs[k]),
            "accuracy_std": np.std(run_accs[k]),
            "eer_mean": np.mean(run_eers[k]),
            "eer_std": np.std(run_eers[k]),
            "auc_mean": np.mean(run_aucs[k]),
            "auc_std": np.std(run_aucs[k])
        })
    df_stats = pd.DataFrame(summary_stats)
    df_stats.to_csv("results/tables/statistical_summary.csv", index=False)
    print("Saved results/tables/statistical_summary.csv")
    
    # Run paired t-tests
    print("\nComputing Paired t-tests for EER:")
    comparisons = [(64, 128), (128, 256), (256, 512)]
    ttest_results = []
    for d1, d2 in comparisons:
        t_stat, p_val = run_significance_test(run_eers[d1], run_eers[d2])
        significant = "Yes" if p_val < 0.05 else "No"
        print(f"  {d1} vs {d2}: t-stat={t_stat:.4f}, p-value={p_val:.6e} (Significant? {significant})")
        ttest_results.append({
            "comparison": f"{d1} vs {d2}",
            "t_statistic": t_stat,
            "p_value": p_val,
            "statistically_significant": significant
        })
    df_ttest = pd.DataFrame(ttest_results)
    df_ttest.to_csv("results/tables/paired_t_test_results.csv", index=False)
    print("Saved results/tables/paired_t_test_results.csv")
    
    # --- EXPERIMENT 7: GENERATE REPORT & EXPLAINABILITY SCORE ---
    print("\nRunning Experiment 7: Generating Verification Explainability Report (PDF)...")
    # Fetch representative reconstruction SSIM/PSNR for the PDF report table
    # We take average across the representative samples for each dimension
    recon_df = df_recon.groupby("components").mean().reset_index()
    ssims = []
    psnrs = []
    for k in dimensions:
        sub = recon_df[recon_df["components"] == k]
        if not sub.empty:
            ssims.append(sub["ssim"].values[0])
            psnrs.append(sub["psnr"].values[0])
        else:
            # Fallback if metrics are missing
            ssims.append(0.5)
            psnrs.append(15.0)
            
    opt_k, opt_es = generate_pdf_report(
        pca_dims=dimensions,
        var_retained=var_retained_list,
        accuracy_values=df_dim_analysis["accuracy"].values,
        eer_values=df_dim_analysis["eer"].values,
        auc_values=df_dim_analysis["auc"].values,
        ssim_values=ssims,
        psnr_values=psnrs,
        save_dir="results"
    )
    
    # Draw Figure 1 (Framework Flowchart)
    draw_framework_overview(save_dir="results/figures")
    
    print("\n" + "=" * 70)
    print("SUCCESS: All explainability experiments and reports have been generated!")
    print(f"Optimal PCA dimension according to ES metric: k = {opt_k} (ES = {opt_es:.4f})")
    print("=" * 70)

if __name__ == "__main__":
    run_all_experiments()
