"""Split up the PC folder into multiple folders to allow parallel training."""
import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)  # TODO: Fix this for debug ...
from helpers import HYPER_DIFF_DIR
from pathlib import Path
import shutil
from progressbar import ProgressBar

if __name__ == "__main__":
    pc_folder_name = "Chair_100000_pc_occ_in_out_True"
    pc_folder_path = HYPER_DIFF_DIR / "data" / "partnet" / "sem_seg_meshes" / pc_folder_name
    n_chunks = 8
    shape_ids_base = set([part.name.split("_")[0] for part in pc_folder_path.glob("*_base.obj.npy")])
    shape_ids_seat = set([part.name.split("_")[0] for part in pc_folder_path.glob("*_seat.obj.npy")])
    shape_ids = list(shape_ids_base.intersection(shape_ids_seat))
    n_shapes = len(shape_ids)
    shapes_per_chunk = n_shapes // n_chunks
    if shapes_per_chunk == 0:
        raise ValueError("More chunks than shapes given.")
    current_chunk = 0
    current_folder_path = pc_folder_path.parent / (pc_folder_name + f"_split_{current_chunk}")
    current_folder_path.mkdir(parents=True, exist_ok=True)
    print("Splitting up the pcs.")
    progress_bar = ProgressBar(maxval=n_shapes).start()
    for idx, shape_id in enumerate(shape_ids):
        if idx == shapes_per_chunk * (current_chunk + 1):
            current_chunk += 1
            current_folder_path = pc_folder_path.parent / (pc_folder_name + f"_split_{current_chunk}")
            current_folder_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(pc_folder_path / (shape_id + "_chair_base.obj.npy"), current_folder_path / (shape_id + "_base.obj.npy"))
        shutil.copy(pc_folder_path / (shape_id + "_chair_seat.obj.npy"), current_folder_path / (shape_id + "_seat.obj.npy"))
        progress_bar.update(idx + 1)
    progress_bar.finish()
    for idx, part in enumerate(pc_folder_path.glob("*_base.obj.npy")):
        if idx == shapes_per_chunk * (current_chunk + 1):
            current_chunk += 1
            current_folder_path = pc_folder_path.parent / (pc_folder_name + f"_split_{current_chunk}")
            current_folder_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(part, current_folder_path / part.name)
        progress_bar.update(idx + 1)
    progress_bar.finish()
    current_chunk = 0
    current_folder_path = pc_folder_path.parent / (pc_folder_name + f"_split_{current_chunk}")
    print("Splitting up the seat pcs.")
    progress_bar = ProgressBar(maxval=n_shapes).start()
    for idx, part in enumerate(pc_folder_path.glob("*_seat.obj.npy")):
        if idx == shapes_per_chunk * (current_chunk + 1):
            current_chunk += 1
            current_folder_path = pc_folder_path.parent / (pc_folder_name + f"_split_{current_chunk}")
        shutil.copy(part, current_folder_path / part.name)
        progress_bar.update(idx + 1)
    progress_bar.finish()
        