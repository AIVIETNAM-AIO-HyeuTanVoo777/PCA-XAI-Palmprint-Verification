import os
import subprocess

def setup_project_structure():
    """Creates output directories and sets up Windows Directory Junctions for data."""
    # Create required subdirectories
    dirs = ["data", "results", "figures", "paper", "src"]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory: {d}")

    # Set up directory junctions if on Windows
    junctions = [
        ("data/IITD", "IITD Dataset"),
        ("data/Tongji", "Tongji Dataset")
    ]
    for target, source in junctions:
        target_norm = os.path.normpath(target)
        source_norm = os.path.normpath(source)
        if not os.path.exists(target_norm):
            try:
                # Use cmd /c mklink /J to create a directory junction
                cmd = f'cmd /c mklink /J "{target_norm}" "{source_norm}"'
                subprocess.run(cmd, shell=True, check=True)
                print(f"Created directory junction: {target_norm} -> {source_norm}")
            except Exception as e:
                print(f"Warning: Failed to create junction for {target_norm}: {e}")
        else:
            print(f"Junction or folder already exists at: {target_norm}")

if __name__ == "__main__":
    setup_project_structure()
