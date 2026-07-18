import os
import hashlib
import time

files = [
    r"results/tables/gabor_xpca_comparison.csv",
    r"results/tables/comprehensive_benchmark.csv",
    r"results/tables/dimension_specific_sota_results.csv",
    r"results/tables/statistical_summary.csv",
    r"results/tables/statistical_significance.csv",
    r"results/tables/open_set_statistical_summary.csv",
    r"results/tables/open_set_statistical_significance.csv",
    r"XPCA-1.tex"
]

for f in files:
    if os.path.exists(f):
        mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(f)))
        with open(f, 'rb') as fh:
            data = fh.read()
            sha = hashlib.sha256(data).hexdigest()
        print(f"{f}:")
        print(f"  MTime: {mtime}")
        print(f"  SHA-256: {sha}")
        print(f"  Size: {len(data)} bytes")
    else:
        print(f"{f}: NOT FOUND")
