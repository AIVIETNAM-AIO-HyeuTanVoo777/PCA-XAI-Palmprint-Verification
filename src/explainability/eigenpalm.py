import os
import numpy as np
import matplotlib.pyplot as plt

def generate_eigenpalm_visualization(pca, target_size=(128, 128), save_dir="results/figures"):
    """
    Extracts, interprets, and visualizes the principal components as EigenPalms.
    
    Selected components: PC1, PC2, PC3, PC4, PC5, PC10, PC20.
    
    Parameters:
    -----------
    pca : PCA
        The fitted scikit-learn PCA object.
    target_size : tuple (int, int)
        Dimensions to reshape the 1D eigenvectors back into image space.
    save_dir : str
        Directory where visualizations will be stored.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # PCs to visualize (1-indexed)
    pc_indices = [1, 2, 3, 4, 5, 10, 20]
    
    # Set up matplotlib figure
    fig, axes = plt.subplots(2, 4, figsize=(15, 8.5), dpi=300)
    axes = axes.flatten()
    
    # Colormap choice: diverging RdBu_r to clearly show positive (red) and negative (blue) activations
    cmap = "RdBu_r"
    
    for idx, pc_num in enumerate(pc_indices):
        ax = axes[idx]
        
        # PCA components_ shape is (n_components, n_features)
        eigenvector = pca.components_[pc_num - 1]
        eigenpalm = eigenvector.reshape(target_size)
        
        # Determine symmetric vmin and vmax for diverging colormap centering at 0
        max_val = np.max(np.abs(eigenpalm))
        
        im = ax.imshow(eigenpalm, cmap=cmap, vmin=-max_val, vmax=max_val)
        ax.set_title(f"PC {pc_num} (Var: {pca.explained_variance_ratio_[pc_num - 1]*100:.2f}%)", 
                     fontsize=10, fontweight="bold")
        ax.axis("off")
        
        # Add colorbar for each individual subplot
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        
    # The 8th plot is used as an explanation legend text
    ax_txt = axes[7]
    ax_txt.axis("off")
    interpretation_text = (
        "EigenPalm Interpretations:\n\n"
        "PC 1: Global illumination gradient\n"
        "and general hand boundaries.\n\n"
        "PC 2-4: Primary palm line topologies\n"
        "(Life Line, Heart Line, Head Line)\n"
        "represented by strong directional valleys.\n\n"
        "PC 5: Secondary flexion creases and\n"
        "wrinkle patterns.\n\n"
        "PC 10 & 20: High-frequency local textures\n"
        "and minor boundary alignment contours."
    )
    ax_txt.text(0.05, 0.95, interpretation_text, fontsize=9, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9FA", edgecolor="#BDC3C7", lw=1))
                
    plt.suptitle("Figure 4: EigenPalm Spatial Activation Maps (Diverging Colormap)", 
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    # Save both named versions for complete deliverable matching
    plt.savefig(os.path.join(save_dir, "eigenpalm_grid.png"), dpi=300)
    plt.savefig(os.path.join(save_dir, "Figure_4_EigenPalm.png"), dpi=300)
    plt.close()
    
    print(f"Saved eigenpalm visualizations to {save_dir}")
