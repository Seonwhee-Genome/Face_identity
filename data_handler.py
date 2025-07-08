import os
import shutil
from pathlib import Path
import re


def data_copy(keyword, relative_filepath, rank=0, determiner=36):
    # Source and destination base paths
    src_base = Path("./testdir")
    dst_base = Path("./yc_data")  # Change this if needed
    
    # Regex to extract number from filename (e.g., 목동힐스테이트_11_좌.jpg → 11)
    number_pattern = re.compile(r"_(\d+)_")

    # Collect all jpgs recursively
    jpg_paths = sorted(src_base.rglob("%s*/%s*.jpg" %(keyword, relative_filepath)))

    for jpg_path in jpg_paths:
        if rank == 0:
            dirname = jpg_path.parent.name
        elif rank == 1:
            dirname = jpg_path.parent.parent.name
        match = number_pattern.search(dirname)
        filename = jpg_path.name
        if not match:
            print(f"⚠️  Skipping {filename} (no number found)")
            continue

        number = int(match.group(1))
        folder_num = number + determiner
        folder_name = f"{folder_num:04d}"
        dst_dir = dst_base / folder_name        
        dst_dir.mkdir(exist_ok=True)
        # Copy file
        dst_file = dst_dir / filename
        shutil.copy(jpg_path, dst_file)
        print(f"✅ Copied {filename} to {folder_name}")


if __name__=="__main__":
    data_copy("목동힐스테이트", "")
    data_copy("목동힐스테이트", "TP/", 1)