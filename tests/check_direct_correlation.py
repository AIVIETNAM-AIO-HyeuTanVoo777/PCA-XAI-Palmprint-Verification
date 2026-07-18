import os
import cv2
import numpy as np

def check_direct():
    f1 = "Tongji Dataset/session1/00001.bmp"
    f2 = "Tongji Dataset/session2/00001.bmp"
    
    img1 = cv2.imread(f1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(f2, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        print("Failed to load images.")
        return
        
    # Resize to 128x128
    img1 = cv2.resize(img1, (128, 128)).astype(np.float32) / 255.0
    img2 = cv2.resize(img2, (128, 128)).astype(np.float32) / 255.0
    
    # Raw cosine similarity
    raw_sim = np.dot(img1.flatten(), img2.flatten()) / (np.linalg.norm(img1) * np.linalg.norm(img2))
    print(f"Direct raw cosine similarity between session1/00001.bmp and session2/00001.bmp: {raw_sim:.4f}")
    
    # Zero-mean cosine similarity
    img1_zm = img1 - np.mean(img1)
    img2_zm = img2 - np.mean(img2)
    zm_sim = np.dot(img1_zm.flatten(), img2_zm.flatten()) / (np.linalg.norm(img1_zm) * np.linalg.norm(img2_zm))
    print(f"Direct zero-mean cosine similarity: {zm_sim:.4f}")
    
    # Now let's compare session1/00001.bmp with session1/00002.bmp (same subject, same session)
    img3 = cv2.imread("Tongji Dataset/session1/00002.bmp", cv2.IMREAD_GRAYSCALE)
    img3 = cv2.resize(img3, (128, 128)).astype(np.float32) / 255.0
    img3_zm = img3 - np.mean(img3)
    same_sess_sim = np.dot(img1_zm.flatten(), img3_zm.flatten()) / (np.linalg.norm(img1_zm) * np.linalg.norm(img3_zm))
    print(f"Zero-mean similarity between session1/00001.bmp and session1/00002.bmp: {same_sess_sim:.4f}")
    
    # Let's compare session1/00001.bmp with session2/04501.bmp (Subject 451, Session 2)
    img4 = cv2.imread("Tongji Dataset/session2/04501.bmp", cv2.IMREAD_GRAYSCALE)
    img4 = cv2.resize(img4, (128, 128)).astype(np.float32) / 255.0
    img4_zm = img4 - np.mean(img4)
    cross_subj_sim = np.dot(img1_zm.flatten(), img4_zm.flatten()) / (np.linalg.norm(img1_zm) * np.linalg.norm(img4_zm))
    print(f"Zero-mean similarity between session1/00001.bmp and session2/04501.bmp: {cross_subj_sim:.4f}")
    
    # Let's find the best match in Session 2 for session1/00001.bmp
    # We scan all session2 files
    best_sim = -1
    best_file = ""
    sess2_files = sorted(glob.glob("Tongji Dataset/session2/*.bmp"))
    for f in sess2_files:
        img_s2 = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img_s2 is None:
            continue
        img_s2 = cv2.resize(img_s2, (128, 128)).astype(np.float32) / 255.0
        img_s2_zm = img_s2 - np.mean(img_s2)
        sim = np.dot(img1_zm.flatten(), img_s2_zm.flatten()) / (np.linalg.norm(img1_zm) * np.linalg.norm(img_s2_zm))
        if sim > best_sim:
            best_sim = sim
            best_file = f
            
    print(f"Best match in Session 2 for session1/00001.bmp: {best_file} with similarity {best_sim:.4f}")

if __name__ == "__main__":
    import glob
    check_direct()
