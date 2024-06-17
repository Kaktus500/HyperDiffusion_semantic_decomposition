from pathlib import Path
import shutil
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent)) # TODO: Fix this for debug ...
from helpers import HYPER_DIFF_DIR

if __name__ == "__main__":
    # Path to the folder containing the .obj files
    obj_folder = HYPER_DIFF_DIR / "data" / "03001627"
    files = list(obj_folder.iterdir())
    nr_files_copied = 0
    # Iterate over all the .obj files in the obj folder
    for idx, obj_file in enumerate(files):
        if not obj_file.is_dir():
            continue
        if not (obj_file / "models" / "model_normalized.obj").exists():
            continue
        if round(idx / len(files), 2) % 0.1 == 0:
            print(f"{idx / len(files) * 100:.2f}% done.")
        shutil.copy(obj_file / "models" / "model_normalized.obj", obj_file.parent / (obj_file.name + ".obj"))
        nr_files_copied += 1

    print(f"Exported {nr_files_copied} .obj files.")