import cv2
import numpy as np

def get_gabor_kernels(ksize=15, sigma=4.0, lambd=8.0, gamma=0.5, orientations=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    Generates a bank of 2D real Gabor kernels.
    """
    kernels = []
    for theta in orientations:
        # cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, psi, ktype)
        # psi=0 for real Gabor filter
        kernel = cv2.getGaborKernel(
            ksize=(ksize, ksize),
            sigma=sigma,
            theta=theta,
            lambd=lambd,
            gamma=gamma,
            psi=0,
            ktype=cv2.CV_32F
        )
        # Normalize the kernel to have zero mean and unit norm to prevent scaling biases
        kernel -= np.mean(kernel)
        kernels.append(kernel)
    return kernels

def extract_gabor_features(images, target_size=(128, 128), ksize=15, sigma=4.0, lambd=8.0, gamma=0.5, downsample_size=(64, 64), orientations=None):
    """
    Convolves each flat image with a Gabor filter bank and returns concatenated, downsampled feature vectors.
    
    Parameters:
    -----------
    images : np.ndarray
        Array of shape (N, D) where each row is a flattened raw image.
    target_size : tuple (int, int)
        Dimensions of the raw image to reshape to.
    ksize : int
        Size of Gabor kernel.
    sigma : float
        Standard deviation of Gaussian envelope.
    lambd : float
        Wavelength of sinusoidal factor.
    gamma : float
        Spatial aspect ratio.
    downsample_size : tuple (int, int)
        Size to downsample each filtered output image to.
    orientations : list of float or None
        List of filter orientations in radians.
        
    Returns:
    --------
    features : np.ndarray
        Array of shape (N, D_gabor) where each row is the concatenated, downsampled Gabor responses.
        If downsample_size=(64, 64) and we have 4 orientations, D_gabor = 4 * 4096 = 16384.
    """
    n_samples = images.shape[0]
    if orientations is None:
        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    kernels = get_gabor_kernels(ksize=ksize, sigma=sigma, lambd=lambd, gamma=gamma, orientations=orientations)
    
    # Calculate Gabor output dimension
    d_gabor = len(orientations) * downsample_size[0] * downsample_size[1]
    gabor_features = np.zeros((n_samples, d_gabor), dtype=np.float32)
    
    print(f"Extracting Gabor features for {n_samples} images (orientations={len(orientations)}, downsample={downsample_size})...")
    
    for idx in range(n_samples):
        img_2d = images[idx].reshape(target_size)
        filtered_list = []
        for kernel in kernels:
            # Apply Gabor filter
            filtered = cv2.filter2D(img_2d, cv2.CV_32F, kernel)
            # Take the absolute value (magnitude response) to get texture energy
            filtered_mag = np.abs(filtered)
            # Downsample to reduce dimensionality and increase translation tolerance
            filtered_resized = cv2.resize(filtered_mag, downsample_size, interpolation=cv2.INTER_LINEAR)
            filtered_list.append(filtered_resized.flatten())
            
        gabor_features[idx] = np.concatenate(filtered_list)
        
    print(f"Gabor feature extraction completed. Output shape: {gabor_features.shape}")
    return gabor_features
