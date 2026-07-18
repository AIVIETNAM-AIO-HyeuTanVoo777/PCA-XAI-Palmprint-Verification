import os
import subprocess

def main():
    print("=" * 60)
    # 1. Run Setup
    print("Phase 1: Project structure setup...")
    subprocess.run("python src/utils.py", shell=True, check=True)
    
    # 2. Run Data Loading Statistics
    print("\nPhase 2: Generating dataset summary...")
    subprocess.run("python src/dataset_loader.py", shell=True, check=True)
    
    # 3. Run PCA Variance Analysis
    print("\nPhase 3: Fitting PCA and analyzing variance...")
    subprocess.run("python src/pca_analysis.py", shell=True, check=True)
    
    # 4. Run Baseline Cosine Verification
    print("\nPhase 4: Running verification baseline...")
    subprocess.run("python src/verification.py", shell=True, check=True)
    
    # 5. Run Reconstruction Analysis
    print("\nPhase 5: Running progressive reconstruction analysis...")
    subprocess.run("python src/reconstruction.py", shell=True, check=True)
    
    # 6. Run Explainability Suite (EigenPalms, Region Occlusion, Component Drop)
    print("\nPhase 6: Running explainability suite...")
    subprocess.run("python src/explainability.py", shell=True, check=True)
    
    print("\n" + "=" * 60)
    print("SUCCESS: All explainability experiments have run successfully!")
    print("All generated results reside in the results/ and figures/ directories.")
    print("=" * 60)

if __name__ == "__main__":
    main()
