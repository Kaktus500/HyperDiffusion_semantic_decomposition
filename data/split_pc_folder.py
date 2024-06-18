"""Split up the PC folder into multiple folders to allow parallel training."""
import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)  # TODO: Fix this for debug ...
from helpers import HYPER_DIFF_DIR
from pathlib import Path
import shutil

if __name__ == "__main__":
    pc_folder_name = "Chair_pc_split_train"
    pc_folder_path = HYPER_DIFF_DIR / "data" / "partnet" / "sem_seg_meshes" / pc_folder_name
    n_chunks = 2
    n_shapes = min(len(list(pc_folder_path.glob("*_base.obj.npy"))), len(list(pc_folder_path.glob("*_seat.obj.npy"))))
    shapes_per_chunk = n_shapes // n_chunks
    if shapes_per_chunk == 0:
        raise ValueError("More chunks than shapes given.")
    current_chunk = 0
    current_folder_path = pc_folder_path.parent / (pc_folder_name + f"_split_{current_chunk}")
    current_folder_path.mkdir(parents=True, exist_ok=True)
    for idx, part in enumerate(pc_folder_path.glob("*_base.obj.npy")):
        if idx == shapes_per_chunk * (current_chunk + 1):
            current_chunk += 1
            current_folder_path = pc_folder_path.parent / (pc_folder_name + f"_split_{current_chunk}")
            current_folder_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(part, current_folder_path / part.name)
    current_chunk = 0
    current_folder_path = pc_folder_path.parent / (pc_folder_name + f"_split_{current_chunk}")
    for idx, part in enumerate(pc_folder_path.glob("*_seat.obj.npy")):
        if idx == shapes_per_chunk * (current_chunk + 1):
            current_chunk += 1
            current_folder_path = pc_folder_path.parent / (pc_folder_name + f"_split_{current_chunk}")
        shutil.copy(part, current_folder_path / part.name)
        