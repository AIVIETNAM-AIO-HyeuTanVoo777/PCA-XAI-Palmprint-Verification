import numpy as np
from sklearn.decomposition import PCA

def fit_pca_model(X_train, n_components=512):
    """
    Fits a PCA model on the training set.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features of shape (N_train, D).
    n_components : int
        Maximum PCA dimensions to retain.
        
    Returns:
    --------
    pca : PCA
        The fitted scikit-learn PCA object.
    """
    # Restrict components to min(samples, features, requested_dims)
    max_comps = min(X_train.shape[0], X_train.shape[1], n_components)
    pca = PCA(n_components=max_comps, random_state=42)
    pca.fit(X_train)
    return pca

def project_embeddings(pca, X, k=None, whiten=False):
    """
    Projects raw features into the PCA subspace.
    
    Parameters:
    -----------
    pca : PCA
        The fitted scikit-learn PCA object.
    X : np.ndarray
        Raw features of shape (N_samples, D).
    k : int or None
        Number of components to retain. If None, uses all available.
    whiten : bool
        If True, whitens the projected features by dividing each component by its standard deviation.
        
    Returns:
    --------
    proj : np.ndarray
        Subspace projected embeddings of shape (N_samples, k).
    """
    proj = pca.transform(X)
    
    if k is not None:
        proj = proj[:, :k]
        
    if whiten:
        # Fetch eigenvalues (explained variances) for the selected k components
        eigenvalues = pca.explained_variance_
        if k is not None:
            eigenvalues = eigenvalues[:k]
        std_devs = np.sqrt(eigenvalues)
        std_devs = np.clip(std_devs, 1e-9, None) # Avoid division by zero
        proj = proj / std_devs
        
    return proj
