import os
import zipfile

def main():
    zip_path = "_paper_resolution/PAPER_RESOLUTION_SAFE_BUNDLE.zip"
    source_dir = "_paper_resolution"
    
    if os.path.exists(zip_path):
        try:
            os.remove(zip_path)
        except OSError:
            pass
            
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                # Skip the zip file itself and python cache files
                if file.endswith('.zip') or file.endswith('.pyc') or file == "zip_bundle.py":
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)
    print(f"Created {zip_path} successfully.")

if __name__ == "__main__":
    main()
