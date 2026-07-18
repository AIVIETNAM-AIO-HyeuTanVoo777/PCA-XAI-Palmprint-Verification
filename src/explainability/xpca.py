import numpy as np

def compute_fisher_scores(train_proj, y_train, eigenvalues, shrinkage=0.1, eps=1e-10):
    """
    Computes global Fisher discriminability scores for each component coordinate.
    
    Parameters:
    -----------
    train_proj : np.ndarray
        PCA projections of training data of shape (N, k).
    y_train : np.ndarray
        Labels of training data of shape (N,).
    eigenvalues : np.ndarray
        Full eigenvalue spectrum of the PCA decomposition.
    shrinkage : float
        Regularization coefficient blending within-class scatter with global eigenvalues.
    eps : float
        Numerical regularization parameter.
        
    Returns:
    --------
    D : np.ndarray
        Fisher discriminability scores of shape (k,).
    """
    k = train_proj.shape[1]
    classes = np.unique(y_train)
    n_classes = len(classes)
    
    class_means = np.zeros((n_classes, k))
    class_vars = np.zeros((n_classes, k))
    
    for idx, c in enumerate(classes):
        mask = (y_train == c)
        z_c = train_proj[mask]
        if len(z_c) > 0:
            class_means[idx] = np.mean(z_c, axis=0)
            if len(z_c) > 1:
                class_vars[idx] = np.var(z_c, axis=0, ddof=1)
            else:
                class_vars[idx] = np.zeros(k)
                
    S_B = np.var(class_means, axis=0)
    S_W = (1.0 - shrinkage) * np.mean(class_vars, axis=0) + shrinkage * eigenvalues[:k]
    
    return S_B / (S_W + eps)

def compute_bootstrap_instability(train_proj, y_train, eigenvalues, shrinkage=0.1, n_iterations=100, seed=42, eps=1e-10):
    """
    Computes statistical stability of the Fisher score using bootstrap resampling.
    
    Parameters:
    -----------
    train_proj : np.ndarray
        PCA projections of training data of shape (N, k).
    y_train : np.ndarray
        Labels of training data of shape (N,).
    eigenvalues : np.ndarray
        Full eigenvalue spectrum of the PCA decomposition.
    shrinkage : float
        Regularization coefficient.
    n_iterations : int
        Number of bootstrap iterations.
    seed : int
        Random seed for reproducibility.
    eps : float
        Numerical regularization parameter.
        
    Returns:
    --------
    N : np.ndarray
        Instability score (coefficient of variation) of shape (k,).
    """
    k = train_proj.shape[1]
    n_samples = len(y_train)
    rng = np.random.default_rng(seed)
    
    bootstrap_fisher = np.zeros((n_iterations, k))
    
    for t in range(n_iterations):
        # Resample with replacement
        indices = rng.choice(n_samples, n_samples, replace=True)
        z_resampled = train_proj[indices]
        y_resampled = y_train[indices]
        
        # Compute Fisher score on resampled dataset
        classes_t, counts_t = np.unique(y_resampled, return_counts=True)
        
        res_means = []
        res_vars = []
        
        for c, count in zip(classes_t, counts_t):
            z_c = z_resampled[y_resampled == c]
            res_means.append(np.mean(z_c, axis=0))
            if count > 1:
                res_vars.append(np.var(z_c, axis=0, ddof=1))
            else:
                res_vars.append(np.zeros(k))
                
        if len(res_means) > 0:
            res_means = np.array(res_means)
            res_vars = np.array(res_vars)
            
            S_B_t = np.var(res_means, axis=0)
            S_W_t = (1.0 - shrinkage) * np.mean(res_vars, axis=0) + shrinkage * eigenvalues[:k]
            bootstrap_fisher[t] = S_B_t / (S_W_t + eps)
        else:
            bootstrap_fisher[t] = np.zeros(k)
            
    # Calculate coefficient of variation: std / (mean + eps)
    std_scores = np.std(bootstrap_fisher, axis=0)
    mean_scores = np.mean(bootstrap_fisher, axis=0)
    
    return std_scores / (mean_scores + eps)

def normalize_utility_scores(scores, method="min-max", temp=1.0, eta=0.1, eps=1e-10):
    """
    Normalizes utility scores using different scaling techniques.
    
    Methods:
    --------
    "min-max" : Standard min-max normalization. Forces lowest score to 0 (+eps), causing coordinate collapse.
    "softmax" : Temperature-controlled softmax. Bounds weights and prevents coordinate collapse.
    "residual": Perturbation around unity: 1 + eta * S_norm.
    """
    if method == "min-max":
        min_val = np.min(scores)
        max_val = np.max(scores)
        return (scores - min_val) / (max_val - min_val + eps)
        
    elif method == "softmax":
        # Shift scores for numerical stability
        shifted = (scores - np.max(scores)) / temp
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores)
        
    elif method == "residual":
        # Compute min-max normalized score first
        min_val = np.min(scores)
        max_val = np.max(scores)
        s_norm = (scores - min_val) / (max_val - min_val + eps)
        return 1.0 + eta * s_norm
        
    elif method in ["sort", "sort_weights"]:
        return scores
        
    else:
        raise ValueError(f"Unknown scaling method: {method}")

class XPCACalibration:
    """
    Calibration class implementing utility-guided diagonal scaling.
    """
    def __init__(self, alpha=0.2, beta=0.6, gamma=0.2, scale_method="min-max", temp=1.0, eta=0.1, bootstrap_iter=100):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.scale_method = scale_method
        self.temp = temp
        self.eta = eta
        self.bootstrap_iter = bootstrap_iter
        self.weights_ = None
        self.V_ = None
        self.D_ = None
        self.N_ = None
        self.sort_indices_ = None
        
    def fit(self, train_proj, y_train, eigenvalues):
        k = train_proj.shape[1]
        
        # 1. Variance score (proportion of explained variance)
        total_var = np.sum(eigenvalues)
        self.V_ = eigenvalues[:k] / total_var
        
        # 2. Fisher discriminability score
        self.D_ = compute_fisher_scores(train_proj, y_train, eigenvalues)
        
        # 3. Bootstrap instability score
        self.N_ = compute_bootstrap_instability(train_proj, y_train, eigenvalues, n_iterations=self.bootstrap_iter)
        
        # 4. Compute additive raw utility score
        raw_utility = self.alpha * self.V_ + self.beta * self.D_ - self.gamma * self.N_
        
        # 5. Normalize weights
        self.weights_ = normalize_utility_scores(raw_utility, method=self.scale_method, temp=self.temp, eta=self.eta)
        
        if self.scale_method == "sort":
            # Descending order based on raw utility
            self.sort_indices_ = np.argsort(raw_utility)[::-1]
        elif self.scale_method == "sort_weights":
            self.sort_indices_ = np.argsort(raw_utility)[::-1]
            self.weights_ = self.V_[self.sort_indices_]
            
        return self
        
    def transform(self, projections):
        """
        Applies diagonal weight matrix to projections, or reorders them if method is 'sort'.
        """
        if self.weights_ is None:
            raise ValueError("XPCACalibration has not been fitted yet.")
            
        if self.scale_method == "sort":
            return projections[:, self.sort_indices_]
        elif self.scale_method == "sort_weights":
            sorted_projections = projections[:, self.sort_indices_]
            return sorted_projections * self.weights_
            
        return projections * self.weights_
