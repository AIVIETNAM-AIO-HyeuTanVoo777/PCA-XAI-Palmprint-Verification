import os
import zipfile

def create_safe_zip(root_dir, zip_filepath):
    # Folders to exclude
    exclude_folders = {
        "IITD Dataset", "Tongji Dataset", "data", ".git", "__pycache__", 
        ".ipynb_checkpoints", ".pytest_cache", "venv", ".venv", "env"
    }
    
    # Files to exclude
    exclude_files = {
        ".env", "PAPER_PROJECT_SAFE_BUNDLE.zip"
    }

    print(f"Creating safe zip archive at: {zip_filepath}")
    
    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(root_dir):
            # Modify dirs in-place to skip excluded folders recursively
            dirs[:] = [d for d in dirs if d not in exclude_folders]
            
            for file in files:
                if file in exclude_files:
                    continue
                if file.endswith(".zip"):
                    continue
                
                full_path = os.path.join(root, file)
                # Compute relative path for zip entry
                rel_path = os.path.relpath(full_path, root_dir)
                
                # Double-check that we are not packaging anything from excluded folders
                path_parts = rel_path.split(os.sep)
                if any(part in exclude_folders for part in path_parts):
                    continue
                
                # Don't add files larger than 10MB just to be safe
                if os.path.getsize(full_path) > 10 * 1024 * 1024:
                    print(f"Skipping large file: {rel_path}")
                    continue
                
                zipf.write(full_path, rel_path)
                
    print("ZIP compression completed successfully!")

if __name__ == "__main__":
    workspace = r"d:\0.Research\AILLLL\PCA-XAI-Palm - Copy (2)"
    zip_out = os.path.join(workspace, "_paper_audit", "PAPER_PROJECT_SAFE_BUNDLE.zip")
    create_safe_zip(workspace, zip_out)
