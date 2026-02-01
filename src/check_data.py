# check_data.py — quick verification that folders & counts are correct
from pathlib import Path
p = Path("data/combined")
print("Root folder:", p.resolve())
classes = [d.name for d in p.iterdir() if d.is_dir()]
classes.sort()
print("Detected class folders:", classes)

for c in classes:
    folder = p / c
    # count files (common image extensions)
    files = []
    for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.tif"):
        files.extend(list(folder.glob(ext)))
    print(f"{c}: {len(files)} files")
