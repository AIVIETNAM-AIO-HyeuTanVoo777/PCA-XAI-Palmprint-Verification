import os
import glob
import cv2
import numpy as np

def inspect():
    sess1_dir = "Tongji Dataset/session1"
    sess2_dir = "Tongji Dataset/session2"
    
    sess1_files = sorted(glob.glob(os.path.join(sess1_dir, "*.bmp")))
    sess2_files = sorted(glob.glob(os.path.join(sess2_dir, "*.bmp")))
    
    for subj in [1, 3, 7]:
        print(f"\nSubject {subj}:")
        # Session 1
        s1_file = sess1_files[(subj-1)*10]
        s1_img = cv2.imread(s1_file, cv2.IMREAD_GRAYSCALE)
        print(f"  Session 1 Image: {os.path.basename(s1_file)}")
        if s1_img is not None:
            print(f"    Shape: {s1_img.shape}, Min: {s1_img.min()}, Max: {s1_img.max()}, Mean: {s1_img.mean():.2f}")
        else:
            print("    Failed to load.")
            
        # Session 2
        s2_file = sess2_files[(subj-1)*10]
        s2_img = cv2.imread(s2_file, cv2.IMREAD_GRAYSCALE)
        print(f"  Session 2 Image: {os.path.basename(s2_file)}")
        if s2_img is not None:
            print(f"    Shape: {s2_img.shape}, Min: {s2_img.min()}, Max: {s2_img.max()}, Mean: {s2_img.mean():.2f}")
        else:
            print("    Failed to load.")

if __name__ == "__main__":
    inspect()
