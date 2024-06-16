import sys
from pathlib import Path

import click

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)  # TODO: Fix this for debug ...
import trimesh

from helpers import HYPER_DIFF_DIR
from siren.experiment_scripts.partnet_pc_generation import normalize_mesh


ply_folder_path = (
    HYPER_DIFF_DIR
    / "siren"
    / "experiment_scripts"
    / "logs"
    / "chair_39767_manifold_ply"
)
category = "Chair"
shape_id = 2803

ground_truth_shape_folder = (
    HYPER_DIFF_DIR / "data" / "partnet" / "sem_seg_meshes" / category
)


complete_mesh = trimesh.Trimesh()
for ply_file in ply_folder_path.glob(f"{shape_id}_*.ply"):
    mesh = trimesh.load_mesh(ply_file)
    complete_mesh += mesh

ground_truth_mesh = trimesh.Trimesh()
for part in ground_truth_shape_folder.glob(f"{shape_id}_*.obj"):
    mesh = trimesh.load_mesh(part)
    ground_truth_mesh += mesh

normalized_ground_truth_mesh, _, _ = normalize_mesh(ground_truth_mesh)

ground_truth_voxel = normalized_ground_truth_mesh.voxelized(0.01)
sdf_voxel = complete_mesh.voxelized(0.01)

# compare count of occupied voxels to check whether the shape is good
# figure out what voxel pitch is good
# make sure coordinate systems are alligned

filled_difference = ground_truth_voxel.filled_count - sdf_voxel.filled_count
print(
    f"Ground truth filled voxels: {ground_truth_voxel.filled_count}, SDF filled voxels: {sdf_voxel.filled_count}"
)
print(f"Filled voxels difference: {filled_difference}")
