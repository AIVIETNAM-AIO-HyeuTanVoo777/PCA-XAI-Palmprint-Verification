import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def analyze_score_distributions(genuine_scores, imposter_scores, save_dir="results"):
    """
    Analyzes matching score distributions and generates score_distribution.png.
    Computes Separation Distance, Bhattacharyya Distance, and Overlap Area.
    
    Parameters:
    -----------
    genuine_scores : np.ndarray
        Genuine matching scores.
    imposter_scores : np.ndarray
        Imposter matching scores.
    save_dir : str
        Directory to save figures and tables.
        
    Returns:
    --------
    stats : dict
        Calculated separation metrics.
    """
    fig_dir = os.path.join(save_dir, "figures")
    tbl_dir = os.path.join(save_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tbl_dir, exist_ok=True)
    
    mu_gen = np.mean(genuine_scores)
    sigma_gen = np.std(genuine_scores)
    
    mu_imp = np.mean(imposter_scores)
    sigma_imp = np.std(imposter_scores)
    
    # 1. Separation Distance (d')
    d_prime = np.abs(mu_gen - mu_imp) / np.sqrt(0.5 * (sigma_gen**2 + sigma_imp**2) + 1e-9)
    
    # 2. Bhattacharyya Distance (DB)
    term1 = 0.25 * np.log(0.25 * (sigma_gen**2 / (sigma_imp**2 + 1e-9) + sigma_imp**2 / (sigma_gen**2 + 1e-9) + 2) + 1e-9)
    term2 = 0.25 * ((mu_gen - mu_imp)**2) / (sigma_gen**2 + sigma_imp**2 + 1e-9)
    b_dist = term1 + term2
    
    # 3. KDE and Overlap Area
    kde_gen = gaussian_kde(genuine_scores)
    kde_imp = gaussian_kde(imposter_scores)
    
    # Cosine similarities are in [-1, 1], but in practice match scores are in [0, 1]
    # We evaluate on 1000 points between min(imposter) and max(genuine)
    eval_min = min(np.min(genuine_scores), np.min(imposter_scores))
    eval_max = max(np.max(genuine_scores), np.max(imposter_scores))
    x_eval = np.linspace(eval_min - 0.1, eval_max + 0.1, 1000)
    
    pdf_gen = kde_gen(x_eval)
    pdf_imp = kde_imp(x_eval)
    
    # Integrate minimum of two KDE PDFs
    overlap_area = np.trapz(np.minimum(pdf_gen, pdf_imp), x_eval)
    
    # 4. Generate Plot
    plt.figure(figsize=(9, 5), dpi=300)
    
    # Histogram of genuine scores
    plt.hist(genuine_scores, bins=40, density=True, alpha=0.3, color="#2CA02C", label="Genuine Histogram")
    # KDE of genuine scores
    plt.plot(x_eval, pdf_gen, color="#2CA02C", lw=2, label="Genuine KDE")
    
    # Histogram of imposter scores
    plt.hist(imposter_scores, bins=80, density=True, alpha=0.3, color="#D62728", label="Imposter Histogram")
    # KDE of imposter scores
    plt.plot(x_eval, pdf_imp, color="#D62728", lw=2, label="Imposter KDE")
    
    # Overlay overlap area shading
    pdf_min = np.minimum(pdf_gen, pdf_imp)
    plt.fill_between(x_eval, pdf_min, color="#9467BD", alpha=0.4, label=f"Overlap Area: {overlap_area:.4f}")
    
    plt.xlabel("Cosine Similarity Score", fontsize=11, fontweight="bold")
    plt.ylabel("Probability Density", fontsize=11, fontweight="bold")
    plt.title(f"Figure 7: Matching Score Distribution Analysis\n(d'={d_prime:.3f}, Bhattacharyya={b_dist:.3f})", 
              fontsize=12, fontweight="bold", pad=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    
    # Save both names
    plt.savefig(os.path.join(fig_dir, "score_distribution.png"), dpi=300)
    plt.savefig(os.path.join(fig_dir, "Figure_7_ScoreDistribution.png"), dpi=300)
    plt.close()
    
    # Save results as text table
    stats_data = {
        "metric": ["separation_distance_d", "bhattacharyya_distance", "overlap_area", "mean_genuine", "std_genuine", "mean_imposter", "std_imposter"],
        "value": [d_prime, b_dist, overlap_area, mu_gen, sigma_gen, mu_imp, sigma_imp]
    }
    pd.DataFrame(stats_data).to_csv(os.path.join(tbl_dir, "distribution_separation_metrics.csv"), index=False)
    
    print(f"Saved distribution analysis to {save_dir}")
    
    return {
        "separation_distance": d_prime,
        "bhattacharyya_distance": b_dist,
        "overlap_area": overlap_area,
        "mu_gen": mu_gen,
        "sigma_gen": sigma_gen,
        "mu_imp": mu_imp,
        "sigma_imp": sigma_imp
    }
