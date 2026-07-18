import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_framework_overview(save_dir="results/figures"):
    """Generates Figure 1: Pipeline Overview flowchart."""
    print("Generating Figure 1: Framework Overview...")
    os.makedirs(save_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 85/10), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    
    # Custom styles
    box_style = dict(boxstyle="round,pad=0.5", facecolor="#EBF3F9", edgecolor="#2C6E91", lw=1.5)
    title_style = dict(fontsize=11, fontweight="bold", color="#1F4E67")
    text_style = dict(fontsize=9, color="#2C3E50")
    
    # 1. Inputs Block
    ax.text(2, 7.5, "IITD Segmented Database\n(Left & Right Palms)", ha="center", va="center", bbox=box_style, **title_style)
    
    # Arrow to Preprocessing
    ax.annotate("", xy=(4.8, 7.5), xytext=(3.6, 7.5), arrowprops=dict(arrowstyle="->", lw=1.5, color="#555555"))
    
    # 2. Preprocessing Block
    ax.text(7.5, 7.5, "Preprocessing Engine\n- Grayscale conversion\n- Spatial resizing (128x128)\n- Individual zero-mean normalization", 
            ha="center", va="center", bbox=box_style, **text_style)
    
    # Arrow to PCA Fitting
    ax.annotate("", xy=(10.2, 7.5), xytext=(11.4, 7.5), arrowprops=dict(arrowstyle="<-", lw=1.5, color="#555555"))
    
    # 3. PCA Embedding Block
    ax.text(13.2, 7.5, "PCA Projection\n- Eigenvectors\n- Variance Analysis", ha="center", va="center", bbox=box_style, **title_style)
    
    # Large bounding box for Multi-Level Explainability
    rect = patches.Rectangle((0.5, 1.2), 15, 4.5, linewidth=2, edgecolor="#8B9B9E", facecolor="#F8FAFC", linestyle="--")
    ax.add_patch(rect)
    ax.text(8, 5.3, "Multi-Level PCA Explainability Suite", ha="center", va="center", fontsize=13, fontweight="bold", color="#2C3E50")
    
    # Sub-blocks under explainability suite
    sub_box_y = 3
    ax.text(2.2, sub_box_y, "Level 1: Global PCA\n- Scree Plots\n- Cumulative Var\n- PCA Dimension Analysis", ha="center", va="center", bbox=box_style, **text_style)
    ax.text(5.9, sub_box_y, "Level 2: EigenPalms\n- PC Semantics\n- Activation maps\n- Visual Interpretability", ha="center", va="center", bbox=box_style, **text_style)
    ax.text(9.9, sub_box_y, "Level 3: Reconstruction\n- Progressive quality\n- SSIM, PSNR, MSE\n- Structural loss", ha="center", va="center", bbox=box_style, **text_style)
    ax.text(13.8, sub_box_y, "Level 4: Decisions\n- Genuine/Imposter Dist\n- Bhattacharyya Distance\n- Explainability Score (ES)", ha="center", va="center", bbox=box_style, **text_style)
    
    # Connect PCA to explainability block
    ax.annotate("", xy=(13.2, 5.7), xytext=(13.2, 6.8), arrowprops=dict(arrowstyle="->", lw=1.5, color="#555555"))
    
    # Final Output Bounding Box
    ax.text(8, 0.5, "Explainable PCA Framework Outputs (CSV, Figures, PDF report, Findings)", 
            ha="center", va="center", bbox=dict(boxstyle="round,pad=0.6", facecolor="#E8F8F5", edgecolor="#117A65", lw=2),
            fontsize=11, fontweight="bold", color="#117A65")
    
    # Arrows from explainability to final output
    ax.annotate("", xy=(8, 0.9), xytext=(8, 1.2), arrowprops=dict(arrowstyle="<-", lw=1.5, color="#555555"))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Figure_1_Framework.png"), bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(save_dir, "framework_overview.png"), bbox_inches="tight", dpi=300)
    plt.close()

def plot_pca_dimension_analysis(pca_dims, accuracies, eers, aucs, save_dir="results/figures"):
    """Generates Figure 2: PCA Dimension Analysis curves."""
    print("Generating Figure 2: PCA Dimension Analysis...")
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(9, 5), dpi=300)
    
    plt.plot(pca_dims, accuracies, marker="o", color="#1F77B4", lw=2, label="Accuracy (Balanced)")
    plt.plot(pca_dims, eers, marker="s", color="#D62728", lw=2, label="Equal Error Rate (EER)")
    plt.plot(pca_dims, aucs, marker="^", color="#2CA02C", lw=2, label="Area Under Curve (AUC)")
    
    plt.xscale("log", base=2)
    plt.xticks(pca_dims, [str(d) for d in pca_dims])
    
    plt.xlabel("PCA Subspace Dimension (k)", fontsize=11, fontweight="bold")
    plt.ylabel("Performance Score", fontsize=11, fontweight="bold")
    plt.title("Figure 2: Palmprint Verification Performance vs. PCA Dimension", fontsize=12, fontweight="bold", pad=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=9, loc="center right")
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, "Figure_2_DimensionAnalysis.png"), dpi=300)
    plt.savefig(os.path.join(save_dir, "pca_dimension_analysis.png"), dpi=300)
    plt.close()

def plot_variance_retention(cumulative_variance, save_dir="results/figures"):
    """Generates Figure 3: Cumulative Explained Variance Retention Curve."""
    print("Generating Figure 3: Variance Retention Curve...")
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(9, 5), dpi=300)
    
    n_components = len(cumulative_variance)
    x = np.arange(1, n_components + 1)
    
    plt.plot(x, cumulative_variance, color="#E67E22", lw=2.5, label="Cumulative Explained Variance")
    
    # Mark Elbow Point (e.g. at k=32/64 where curvature turns)
    elbow_x = 32
    elbow_y = cumulative_variance[elbow_x - 1]
    plt.plot(elbow_x, elbow_y, "ro", markersize=7)
    plt.annotate(f"Elbow Point (k={elbow_x}, {elbow_y*100:.1f}%)", 
                 xy=(elbow_x, elbow_y), 
                 xytext=(elbow_x + 30, elbow_y - 0.15),
                 arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
                 fontsize=9, fontweight="bold")
                 
    # Mark Saturation region (e.g. cumulative variance crosses 95%)
    idx_95 = np.argmax(cumulative_variance >= 0.95)
    sat_x = idx_95 + 1
    sat_y = cumulative_variance[idx_95]
    plt.axvline(x=sat_x, color="#2980B9", linestyle=":", lw=1.2)
    plt.plot(sat_x, sat_y, "bo", markersize=7)
    plt.annotate(f"95% Saturation (k={sat_x})", 
                 xy=(sat_x, sat_y), 
                 xytext=(sat_x + 40, sat_y - 0.08),
                 arrowprops=dict(facecolor='blue', shrink=0.08, width=1, headwidth=6),
                 fontsize=9, fontweight="bold")
                 
    plt.xlabel("Number of Principal Components", fontsize=11, fontweight="bold")
    plt.ylabel("Cumulative Explained Variance Ratio", fontsize=11, fontweight="bold")
    plt.title("Figure 3: Cumulative Explained Variance Curve", fontsize=12, fontweight="bold", pad=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, "Figure_3_VarianceRetention.png"), dpi=300)
    plt.savefig(os.path.join(save_dir, "variance_analysis.png"), dpi=300)
    plt.close()

def plot_component_importance(eigenvalues, save_dir="results/figures"):
    """Generates Figure 6: Component Importance (Scree Plot & Top-20 Importance)."""
    print("Generating Figure 6: Component Importance...")
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # 1. Scree Plot (all components)
    x_all = np.arange(1, len(eigenvalues) + 1)
    ev_ratio = eigenvalues / np.sum(eigenvalues)
    
    axes[0].plot(x_all, ev_ratio, color="#2C3E50", lw=2, label="Individual Explained Var")
    axes[0].set_xlabel("Principal Component Index", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Explained Variance Ratio", fontsize=10, fontweight="bold")
    axes[0].set_title("A. Complete Scree Plot (All Components)", fontsize=11, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend(fontsize=9)
    
    # 2. Bar plot (Top-20 PC Importance)
    top_n = min(20, len(eigenvalues))
    x_top = np.arange(1, top_n + 1)
    
    axes[1].bar(x_top, ev_ratio[:top_n], color="#2980B9", edgecolor="#1A5276", alpha=0.85, label="PC Importance")
    axes[1].set_xticks(x_top)
    axes[1].set_xlabel("Principal Component Index", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Explained Variance Ratio", fontsize=10, fontweight="bold")
    axes[1].set_title("B. Top-20 Component Variance Contribution", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend(fontsize=9)
    
    plt.suptitle("Figure 6: Component Importance Analysis (Scree & Top-20 Contribution)", 
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, "Figure_6_ComponentImportance.png"), dpi=300)
    plt.savefig(os.path.join(save_dir, "component_importance.png"), dpi=300)
    plt.close()
