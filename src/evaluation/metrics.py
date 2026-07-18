import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy.stats import ttest_rel

def compute_eer(genuine_scores, imposter_scores):
    """
    Computes Equal Error Rate (EER) and the corresponding threshold.
    """
    y_true = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(imposter_scores)])
    y_scores = np.concatenate([genuine_scores, imposter_scores])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    
    # EER is where fpr == fnr
    idx = np.nanargmin(np.absolute(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    threshold = thresholds[idx]
    
    return eer, threshold, fpr, fnr

def evaluate_verification(train_proj, y_train, test_proj, y_test, n_classes):
    """
    Evaluates verification performance using template matching with Cosine Similarity.
    
    Parameters:
    -----------
    train_proj : np.ndarray
        PCA-projected training embeddings.
    y_train : np.ndarray
        Training labels.
    test_proj : np.ndarray
        PCA-projected test embeddings.
    y_test : np.ndarray
        Test labels.
    n_classes : int
        Number of classes.
        
    Returns:
    --------
    metrics : dict
        A dictionary containing Accuracy, EER, AUC, FAR, FRR, genuine_scores, and imposter_scores.
    """
    # 1. Compute templates (mean training vector per class)
    k = train_proj.shape[1]
    templates = []
    for c in range(n_classes):
        class_mask = (y_train == c)
        if np.sum(class_mask) == 0:
            templates.append(np.zeros(k))
        else:
            templates.append(np.mean(train_proj[class_mask], axis=0))
    templates = np.array(templates)
    
    # 2. Normalize vectors for Cosine Similarity
    test_norm = test_proj / np.clip(np.linalg.norm(test_proj, axis=1, keepdims=True), 1e-9, None)
    temp_norm = templates / np.clip(np.linalg.norm(templates, axis=1, keepdims=True), 1e-9, None)
    
    # 3. Compute similarity matrix
    sim_matrix = test_norm @ temp_norm.T # shape: (N_test, C)
    
    # 4. Extract genuine and imposter scores
    genuine_scores = sim_matrix[np.arange(len(y_test)), y_test]
    
    mask = np.ones_like(sim_matrix, dtype=bool)
    mask[np.arange(len(y_test)), y_test] = False
    imposter_scores = sim_matrix[mask]
    
    # 5. Compute EER and metrics
    eer, threshold, fpr, fnr = compute_eer(genuine_scores, imposter_scores)
    
    y_true = np.concatenate([np.ones_like(genuine_scores), np.zeros_like(imposter_scores)])
    y_scores = np.concatenate([genuine_scores, imposter_scores])
    fpr_auc, tpr_auc, _ = roc_curve(y_true, y_scores, pos_label=1)
    roc_auc = auc(fpr_auc, tpr_auc)
    
    accuracy = 1.0 - eer # Balanced Accuracy
    
    return {
        "accuracy": accuracy,
        "eer": eer,
        "auc": roc_auc,
        "far": eer, # At EER threshold, FAR = EER
        "frr": eer, # At EER threshold, FRR = EER
        "threshold": threshold,
        "genuine_scores": genuine_scores,
        "imposter_scores": imposter_scores,
        "fpr_curve": fpr_auc,
        "tpr_curve": tpr_auc
    }

def run_significance_test(results_dim1, results_dim2):
    """
    Performs a Relational Paired t-test comparing verification performance (EER) of two dimensions.
    
    Parameters:
    -----------
    results_dim1 : list or np.ndarray
        List of EER values from multiple runs for dimension 1.
    results_dim2 : list or np.ndarray
        List of EER values from multiple runs for dimension 2.
        
    Returns:
    --------
    t_stat : float
        t-statistic.
    p_value : float
        Two-tailed p-value.
    """
    t_stat, p_val = ttest_rel(results_dim1, results_dim2)
    return t_stat, p_val
