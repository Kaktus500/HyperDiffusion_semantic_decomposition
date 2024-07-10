"""Split up the PC folder into multiple folders to allow parallel training."""
import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)  # TODO: Fix this for debug ...
import shutil
from pathlib import Path
from typing import Set, Union

import numpy as np
from progressbar import ProgressBar

from helpers import HYPER_DIFF_DIR


def split_pc_folder(shape_name: str, n_chunks: int, pc_folder_path: Path, part_names: Set[str], fill_missing_part_with_empty_pc: Union[str, None] = None):
    """Split a point cloud folder into a given number of chunks."""
    part_shape_ids = {}
    shape_ids = set()
    for i, part_name in enumerate(part_names):
        part_shape_ids[part_name] = set([part.name.split("_")[0] for part in pc_folder_path.glob(f"*_{part_name}.obj.npy")])
        if fill_missing_part_with_empty_pc is not None and fill_missing_part_with_empty_pc == part_name:
            continue
        if i == 0:
            shape_ids = part_shape_ids[part_name]
            continue
        shape_ids.intersection_update(part_shape_ids[part_name])
    # shape_ids_base = set([part.name.split("_")[0] for part in pc_folder_path.glob("*_base.obj.npy")])
    # shape_ids_seat = set([part.name.split("_")[0] for part in pc_folder_path.glob("*_seat.obj.npy")])
    # shape_ids_back = set([part.name.split("_")[0] for part in pc_folder_path.glob("*_back.obj.npy")])
    # shape_ids_arm = set([part.name.split("_")[0] for part in pc_folder_path.glob("*_arm.obj.npy")])

    # find the shapes that have all components
    shape_ids_missing = set()
    if fill_missing_part_with_empty_pc is not None:
        shape_ids_missing = shape_ids - part_shape_ids[fill_missing_part_with_empty_pc]
    shape_ids = list(shape_ids - shape_ids_missing)
    shape_ids_missing = list(shape_ids_missing)
    n_shapes = len(shape_ids) + len(shape_ids_missing)
    shapes_per_chunk = n_shapes // n_chunks
    if shapes_per_chunk == 0:
        raise ValueError("More chunks than shapes given.")
    current_chunk = 0
    current_folder_path = pc_folder_path.parent / (pc_folder_name + "_full_arms" + f"_split_{current_chunk}")
    current_folder_path.mkdir(parents=True, exist_ok=True)
    print("Splitting up the pcs.")
    progress_bar = ProgressBar(maxval=n_shapes).start()
    # first transfer all shapes for which all parts exist
    for idx, shape_id in enumerate(shape_ids):
        if idx == shapes_per_chunk * (current_chunk + 1):
            current_chunk += 1
            current_folder_path = pc_folder_path.parent / (pc_folder_name + "_full_arms" + f"_split_{current_chunk}")
            current_folder_path.mkdir(parents=True, exist_ok=True)
        for part_name in part_names:
            shutil.copy(pc_folder_path / (shape_id + f"_{shape_name}_{part_name}.obj.npy"), current_folder_path / (shape_id + f"_{part_name}.obj.npy"))
        progress_bar.update(idx + 1)
    idx_old = shapes_per_chunk * (current_chunk + 1)
    # then transfer all shapes for which one part is missing
    for idx, shape_id in enumerate(shape_ids_missing):
        if idx_old + idx == shapes_per_chunk * (current_chunk + 1):
            current_chunk += 1
            current_folder_path = pc_folder_path.parent / (pc_folder_name + f"_full_zeroed_{fill_missing_part_with_empty_pc}" + f"_split_{current_chunk}")
            current_folder_path.mkdir(parents=True, exist_ok=True)
        for part_name in part_names:
            if fill_missing_part_with_empty_pc == part_name:
                # fill all missing parts with empty point clouds
                point_cloud = np.random.uniform(-0.5, 0.5, size=(200000, 3))
                occupancies = np.zeros((200000, 1))
                point_cloud = np.hstack((point_cloud, occupancies))
                np.save(current_folder_path / (shape_id + f"_{part_name}.obj.npy"), point_cloud)
            else:
                shutil.copy(pc_folder_path / (shape_id + f"_{shape_name}_{part_name}.obj.npy"), current_folder_path / (shape_id + f"_{part_name}.obj.npy"))
        progress_bar.update(progress_bar.currval + 1)

    progress_bar.finish()

if __name__ == "__main__":
    pc_folder_name = "Chair_100000_pc_occ_in_out_True"
    pc_folder_path = HYPER_DIFF_DIR / "data" / "partnet" / "sem_seg_meshes" / pc_folder_name
    n_chunks = 8
    split_pc_folder("chair", n_chunks, pc_folder_path, {"base", "seat", "back", "arm"}, fill_missing_part_with_empty_pc="arm")
    # shape_ids_base = set([part.name.split("_")[0] for part in pc_folder_path.glob("*_base.obj.npy")])
    # shape_ids_seat = set([part.name.split("_")[0] for part in pc_folder_path.glob("*_seat.obj.npy")])
    # shape_ids_back = set([part.name.split("_")[0] for part in pc_folder_path.glob("*_back.obj.npy")])
    # shape_ids_arm = set([part.name.split("_")[0] for part in pc_folder_path.glob("*_arm.obj.npy")])

    # # find the shapes that have all components
    # shape_ids = list(shape_ids_base.intersection(shape_ids_seat, shape_ids_back, shape_ids_arm))
    # n_shapes = len(shape_ids)
    # shapes_per_chunk = n_shapes // n_chunks
    # if shapes_per_chunk == 0:
    #     raise ValueError("More chunks than shapes given.")
    # current_chunk = 0
    # current_folder_path = pc_folder_path.parent / (pc_folder_name + "_full" + f"_split_{current_chunk}")
    # current_folder_path.mkdir(parents=True, exist_ok=True)
    # print("Splitting up the pcs.")
    # progress_bar = ProgressBar(maxval=n_shapes).start()
    # for idx, shape_id in enumerate(shape_ids):
    #     if idx == shapes_per_chunk * (current_chunk + 1):
    #         current_chunk += 1
    #         current_folder_path = pc_folder_path.parent / (pc_folder_name + "_full" + f"_split_{current_chunk}")
    #         current_folder_path.mkdir(parents=True, exist_ok=True)
    #     shutil.copy(pc_folder_path / (shape_id + "_chair_base.obj.npy"), current_folder_path / (shape_id + "_base.obj.npy"))
    #     shutil.copy(pc_folder_path / (shape_id + "_chair_seat.obj.npy"), current_folder_path / (shape_id + "_seat.obj.npy"))
    #     shutil.copy(pc_folder_path / (shape_id + "_chair_back.obj.npy"), current_folder_path / (shape_id + "_back.obj.npy"))
    #     shutil.copy(pc_folder_path / (shape_id + "_chair_arm.obj.npy"), current_folder_path / (shape_id + "_arm.obj.npy"))
    #     progress_bar.update(idx + 1)
    # progress_bar.finish()
        