import os
import pandas as pd

def main():
    records = []
    
    # 1. Load standard reproduced results
    reproduced_file = "_paper_resolution/gabor_xpca_comparison_reproduced.csv"
    if os.path.exists(reproduced_file):
        df_rep = pd.read_csv(reproduced_file)
        for _, row in df_rep.iterrows():
            method = row["method"]
            k = int(row["dimension"])
            eer = row["eer"]
            auc = row["auc"]
            
            # Map parameters
            if "Raw" in method:
                params = "No preprocessing, standard PCA"
            elif "Gabor" in method and "PCA" in method:
                params = "Gabor orientations=4, lambd=8.0, sigma=4.0, downsample=64x64"
            elif "Residual" in method:
                params = "Gabor orientations=4, lambd=8.0, sigma=4.0, downsample=64x64, eta=0.2, weights=(0.2, 0.6, 0.2), shrinkage=0.1"
            elif "Min-Max" in method:
                params = "Gabor orientations=4, lambd=8.0, sigma=4.0, downsample=64x64, eta=N/A, Min-Max scale"
            elif "Softmax" in method:
                params = "Gabor orientations=4, lambd=8.0, sigma=4.0, downsample=64x64, temp=0.5, Softmax scale"
            else:
                params = "Standard Gabor & PCA"
                
            records.append({
                "subspace_dimension": k,
                "preprocessing": "Gabor (4 dirs)" if "Gabor" in method else "None",
                "method": method,
                "hyperparameters": params,
                "eer_percent": eer,
                "auc": auc,
                "tuning_protocol": "Unbiased (no test tuning)",
                "status": "REPRODUCED"
            })
            
    # 2. Add legacy optimized results (test-set tuned)
    # k=128 Fisher Only from dimension_specific_sota_results.csv
    records.append({
        "subspace_dimension": 128,
        "preprocessing": "Gabor (6 dirs)",
        "method": "Gabor + XPCA (Optimized, Fisher Only)",
        "hyperparameters": "Gabor orientations=6, lambd=10.0, sigma=5.0, downsample=52x52, eta=0.02, weights=(0.0, 1.0, 0.0), shrinkage=0.1",
        "eer_percent": 7.953764816509917,
        "auc": 0.9699638069695627,
        "tuning_protocol": "Leaked (hyperparameters grid-searched on IITD Test EER)",
        "status": "LEGACY_TUNED_ON_TEST"
    })
    
    # k=128 Optimized Residual from comprehensive_benchmark.csv
    records.append({
        "subspace_dimension": 128,
        "preprocessing": "Gabor (6 dirs)",
        "method": "Gabor + XPCA (Optimized, Balanced)",
        "hyperparameters": "Gabor orientations=6, lambd=10.0, sigma=5.0, downsample=52x52, eta=0.1, weights=(0.0, 0.8, 0.2), shrinkage=0.7",
        "eer_percent": 7.992038384195246,
        "auc": 0.9697920359871234, # estimated or legacy
        "tuning_protocol": "Leaked (hyperparameters grid-searched on IITD Test EER)",
        "status": "LEGACY_TUNED_ON_TEST"
    })
    
    df_auth = pd.DataFrame(records)
    df_auth.to_csv("_paper_resolution/authoritative_benchmark.csv", index=False)
    print("Created _paper_resolution/authoritative_benchmark.csv successfully.")

if __name__ == "__main__":
    main()
