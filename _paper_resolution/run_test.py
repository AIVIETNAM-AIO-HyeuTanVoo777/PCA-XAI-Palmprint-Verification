import os
import sys
import numpy as np
from sklearn.decomposition import PCA

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing.loader import load_iitd_dataset
from preprocessing.gabor import extract_gabor_features
from explainability.xpca import XPCACalibration
from evaluation.metrics import evaluate_verification

print("Loading dataset...")
X_train_raw, y_train, X_test_raw, y_test, n_classes, _ = load_iitd_dataset(
    data_dir="data/IITD", max_subjects=None, target_size=(128, 128), zero_mean_samples=True
)

X_train_no_zm, _, X_test_no_zm, _, _, _ = load_iitd_dataset(
    data_dir="data/IITD", max_subjects=None, target_size=(128, 128), zero_mean_samples=False
)

print("Extracting standard Gabor features...")
X_train_gabor = extract_gabor_features(X_train_no_zm, target_size=(128, 128), downsample_size=(64, 64))
X_test_gabor = extract_gabor_features(X_test_no_zm, target_size=(128, 128), downsample_size=(64, 64))

X_train_gabor = X_train_gabor - np.mean(X_train_gabor, axis=1, keepdims=True)
X_test_gabor = X_test_gabor - np.mean(X_test_gabor, axis=1, keepdims=True)

print("Fitting PCA on Gabor features...")
pca_gabor = PCA(n_components=512, random_state=42)
pca_gabor.fit(X_train_gabor)

k = 128
tr_gabor = pca_gabor.transform(X_train_gabor)[:, :k]
te_gabor = pca_gabor.transform(X_test_gabor)[:, :k]

print("Evaluating Gabor+PCA Baseline...")
res_gab_pca = evaluate_verification(tr_gabor, y_train, te_gabor, y_test, n_classes)
print(f"Gabor+PCA Baseline EER: {res_gab_pca['eer'] * 100}% (Raw: {res_gab_pca['eer']})")

print("Fitting and evaluating Gabor+XPCA (Residual) at k=128, eta=0.2...")
xpca_gab_res = XPCACalibration(scale_method="residual", eta=0.2, bootstrap_iter=100)
xpca_gab_res.fit(tr_gabor, y_train, pca_gabor.explained_variance_[:k])
tr_gab_res = xpca_gab_res.transform(tr_gabor)
te_gab_res = xpca_gab_res.transform(te_gabor)
res_gab_xpca_res = evaluate_verification(tr_gab_res, y_train, te_gab_res, y_test, n_classes)
print(f"Gabor+XPCA Residual EER: {res_gab_xpca_res['eer'] * 100}% (Raw: {res_gab_xpca_res['eer']})")
