import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)  # TODO: Fix this for debug ...
import json

import numpy as np
import trimesh
from progressbar import ProgressBar

from helpers import HYPER_DIFF_DIR


def compute_quality_metrics(
    ply_folder_path: Path,
    ground_truth_shape_folder: Path,
    shape_ids: List[str],
) -> Dict[str, Dict[str, float]]:
    metrics = defaultdict(dict)
    progress_bar = ProgressBar(maxval=len(shape_ids)).start()
    for idx, shape_id in enumerate(shape_ids):
        sdf_mesh = trimesh.Trimesh()
        for ply_file in ply_folder_path.glob(f"*{shape_id}_*.ply"):
            mesh = trimesh.load_mesh(ply_file)
            sdf_mesh += mesh

        ground_truth_mesh = trimesh.Trimesh()
        for part in ground_truth_shape_folder.glob(f"{shape_id}_*.obj"):
            mesh = trimesh.load_mesh(part)
            ground_truth_mesh += mesh

        # roughly check whether the meshes are somewhat aligned

        bounds_difference = ground_truth_mesh.bounds - sdf_mesh.bounds

        ground_truth_voxel = ground_truth_mesh.voxelized(0.01)
        sdf_voxel = sdf_mesh.voxelized(0.01)

        # compare count of occupied voxels to check whether the shape is good

        # filled difference value of 0.05 is a first good threshold for filtering results
        filled_difference = abs(
            ground_truth_voxel.filled_count - sdf_voxel.filled_count
        )
        metrics[shape_id]["bounds_difference_norm"] = np.linalg.norm(
            bounds_difference, axis=1
        ).tolist()
        metrics[shape_id]["filled_difference"] = filled_difference
        metrics[shape_id]["filled_difference_fraction"] = (
            filled_difference / ground_truth_voxel.filled_count
        )
        progress_bar.update(idx + 1)
    progress_bar.finish()
    return metrics


if __name__ == "__main__":
    ply_folder_path = (
        HYPER_DIFF_DIR
        / "siren"
        / "experiment_scripts"
        / "logs"
        / "chair_base_seat_ply"
    )
    category = "Chair"
    shape_ids = list(ply_folder_path.glob("*.ply"))
    shape_ids = list(set([file.stem.split("_")[1] for file in shape_ids]))

    ground_truth_shape_folder = (
        HYPER_DIFF_DIR / "data" / "partnet" / "sem_seg_meshes" / category
    )

    output_metrics_folder = (
        HYPER_DIFF_DIR
        / "siren"
        / "experiment_scripts"
        / "logs"
        / "chair_base_seat_ply"
        / "metrics.json"
    )
    metrics = compute_quality_metrics(
        ply_folder_path, ground_truth_shape_folder, shape_ids
    )

    with open(output_metrics_folder, "w") as f:
        json.dump(metrics, f)
