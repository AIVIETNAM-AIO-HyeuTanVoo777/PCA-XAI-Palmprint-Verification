import os
import glob
import cv2
import numpy as np

def load_iitd_dataset(data_dir="data/IITD", max_subjects=None, target_size=(128, 128), zero_mean_samples=True, random_split=False, seed=42):
    """
    Loads the IIT Delhi (IITD) Touchless Palmprint Database.
    
    Splits: First 3 sorted images per palm class -> Train, remaining -> Test.
    Identity mapping: Left and Right hands are treated as separate palms/classes.
    
    Parameters:
    -----------
    data_dir : str
        Path to the IITD dataset directory.
    max_subjects : int or None
        If specified, loads only up to this number of unique subjects (each has Left and Right hands).
    target_size : tuple (int, int)
        Image resizing dimensions.
    zero_mean_samples : bool
        If True, applies individual zero-mean centering to each sample image.
    random_split : bool
        If True, shuffles the files before partitioning.
    seed : int
        Seed for the random generator to guarantee reproducibility.
        
    Returns:
    --------
    X_train : np.ndarray
        Training feature matrix of shape (N_train, D).
    y_train : np.ndarray
        Training labels of shape (N_train,).
    X_test : np.ndarray
        Testing feature matrix of shape (N_test, D).
    y_test : np.ndarray
        Testing labels of shape (N_test,).
    n_classes : int
        Total number of unique palm classes loaded.
    unique_subjects : int
        Total number of unique human subjects loaded.
    """
    print(f"Loading IITD dataset from {data_dir} (max_subjects={max_subjects}, random_split={random_split}, seed={seed})...")
    seg_dir = os.path.join(data_dir, "Segmented")
    left_dir = os.path.join(seg_dir, "Left")
    right_dir = os.path.join(seg_dir, "Right")
    
    # Check if directories exist (handle paths relative to execution dir)
    if not os.path.exists(left_dir) or not os.path.exists(right_dir):
        # Try fallbacks
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        left_dir = os.path.join(base_dir, "IITD Dataset", "Segmented", "Left")
        right_dir = os.path.join(base_dir, "IITD Dataset", "Segmented", "Right")
        if not os.path.exists(left_dir) or not os.path.exists(right_dir):
            raise FileNotFoundError(f"IITD Segmented Left/Right directories not found under {seg_dir}")
            
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    left_files = glob.glob(os.path.join(left_dir, "*.bmp"))
    right_files = glob.glob(os.path.join(right_dir, "*.bmp"))
    
    def group_by_palm(files, side_label):
        palm_groups = {}
        for f in files:
            basename = os.path.basename(f)
            parts = basename.split("_")
            if len(parts) < 2:
                continue
            subj_str = parts[0]
            try:
                subj_id = int(subj_str)
            except ValueError:
                continue
                
            palm_key = f"{subj_id}_{side_label}"
            if palm_key not in palm_groups:
                palm_groups[palm_key] = []
            palm_groups[palm_key].append(f)
        return palm_groups

    left_groups = group_by_palm(left_files, "L")
    right_groups = group_by_palm(right_files, "R")
    
    all_groups = {}
    all_groups.update(left_groups)
    all_groups.update(right_groups)
    
    # Sort keys to ensure deterministic mapping
    sorted_keys = sorted(all_groups.keys(), key=lambda x: (int(x.split("_")[0]), x.split("_")[1]))
    
    # Filter subjects if max_subjects is specified
    if max_subjects is not None:
        unique_subjs = sorted(list(set(int(k.split("_")[0]) for k in sorted_keys)))
        allowed_subjs = unique_subjs[:max_subjects]
        sorted_keys = [k for k in sorted_keys if int(k.split("_")[0]) in allowed_subjs]
        
    class_map = {k: i for i, k in enumerate(sorted_keys)}
    n_classes = len(class_map)
    
    for palm_key in sorted_keys:
        files = all_groups[palm_key]
        class_idx = class_map[palm_key]
        
        # Sort files to guarantee order (e.g. 001_1.bmp, 001_2.bmp ...)
        def get_sample_id(f):
            basename = os.path.basename(f)
            parts = basename.split("_")
            if len(parts) >= 2:
                try:
                    return int(parts[1].split(".")[0])
                except ValueError:
                    pass
            return 999
            
        files_sorted = sorted(files, key=get_sample_id)
        
        if random_split:
            # We shuffle with a local random state to ensure reproducibility per class
            rng = np.random.default_rng(seed + class_idx)
            files_shuffled = files_sorted.copy()
            rng.shuffle(files_shuffled)
            train_files = files_shuffled[:3]
            test_files = files_shuffled[3:]
        else:
            train_files = files_sorted[:3]
            test_files = files_sorted[3:]
        
        for f in train_files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            img_flat = img_resized.astype(np.float32) / 255.0
            
            if zero_mean_samples:
                img_flat = img_flat - np.mean(img_flat)
                
            X_train.append(img_flat.flatten())
            y_train.append(class_idx)
            
        for f in test_files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            img_flat = img_resized.astype(np.float32) / 255.0
            
            if zero_mean_samples:
                img_flat = img_flat - np.mean(img_flat)
                
            X_test.append(img_flat.flatten())
            y_test.append(class_idx)
            
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)
    
    unique_subjects = len(set(k.split("_")[0] for k in sorted_keys))
    
    print(f"IITD Loaded: Train={X_train.shape}, Test={X_test.shape}, Classes={n_classes}, Subjects={unique_subjects}")
    return X_train, y_train, X_test, y_test, n_classes, unique_subjects
