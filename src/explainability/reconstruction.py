import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_func

def compute_psnr(original, reconstructed):
    """Computes Peak Signal-to-Noise Ratio (PSNR) for normalized images [0, 1]."""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))

def run_reconstruction_analysis(pca, X_samples, target_size=(128, 128), save_dir="results"):
    """
    Reconstructs selected samples using 16, 32, 64, 128, 256, 512 components.
    Computes MSE, PSNR, and SSIM for each reconstruction and saves reports and plots.
    
    Parameters:
    -----------
    pca : PCA
        Fitted scikit-learn PCA object.
    X_samples : np.ndarray
        Representative samples to reconstruct, shape (N_samples, D).
    target_size : tuple (int, int)
        Reshaped dimensions of the image.
    save_dir : str
        Base directory to save results.
    """
    fig_dir = os.path.join(save_dir, "figures")
    tbl_dir = os.path.join(save_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tbl_dir, exist_ok=True)
    
    k_values = [16, 32, 64, 128, 256, 512]
    metrics = []
    
    n_samples = len(X_samples)
    
    # Grid plot: n_samples rows, 7 columns (Original + 6 reconstruction levels)
    fig, axes = plt.subplots(n_samples, 7, figsize=(16, 2.5 * n_samples), dpi=300)
    if n_samples == 1:
        axes = np.expand_dims(axes, axis=0)
        
    for sample_idx in range(n_samples):
        sample = X_samples[sample_idx]
        orig_img = sample.reshape(target_size)
        
        # Render original
        axes[sample_idx, 0].imshow(orig_img, cmap="gray")
        axes[sample_idx, 0].set_title(f"Sample {sample_idx+1}\nOriginal", fontsize=9, fontweight="bold")
        axes[sample_idx, 0].axis("off")
        
        for k_idx, k in enumerate(k_values):
            # Limit components to maximum available in the PCA model
            k_val = min(k, pca.n_components_)
            
            # Project and reconstruct
            proj = pca.transform(sample.reshape(1, -1))[:, :k_val]
            components = pca.components_[:k_val]
            mean = pca.mean_
            
            reconstructed = proj @ components + mean
            reconstructed = np.clip(reconstructed[0], 0.0, 1.0)
            
            recon_img = reconstructed.reshape(target_size)
            
            # Compute structural metrics
            mse = np.mean((sample - reconstructed) ** 2)
            psnr = compute_psnr(sample, reconstructed)
            ssim = ssim_func(orig_img, recon_img, data_range=1.0)
            
            metrics.append({
                "sample_index": sample_idx + 1,
                "components": k,
                "mse": mse,
                "psnr": psnr,
                "ssim": ssim
            })
            
            # Display image
            ax = axes[sample_idx, k_idx + 1]
            ax.imshow(recon_img, cmap="gray")
            ax.set_title(f"k={k}\nSSIM:{ssim:.3f}", fontsize=8)
            ax.axis("off")
            
    plt.suptitle("Figure 5: Progressive PCA Palmprint Reconstruction Grid", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()
    
    # Save visualizations
    plt.savefig(os.path.join(fig_dir, "reconstruction_quality.png"), dpi=300)
    plt.savefig(os.path.join(fig_dir, "Figure_5_Reconstruction.png"), dpi=300)
    plt.close()
    
    # Export metrics as CSV
    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(save_dir, "reconstruction_metrics.csv"), index=False)
    df.to_csv(os.path.join(tbl_dir, "reconstruction_metrics.csv"), index=False)
    
    print(f"Saved reconstruction analysis to {save_dir}")
    return df
