import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)  # TODO: Fix this for debug ...
import json

import numpy as np
from helpers import HYPER_DIFF_DIR
import trimesh

pc_path = HYPER_DIFF_DIR / "data" / "chair_100000_pc_occ_in_out_True"

mesh = trimesh.load_mesh(HYPER_DIFF_DIR / "data" / "chair" / "chair_arm_filled.obj")
# mesh_nf = trimesh.load_mesh(
#     HYPER_DIFF_DIR
#     / "siren"
#     / "experiment_scripts"
#     / "logs"
#     / "test_combined_mesh_ply"
#     / "occ_chair_arm_filled_jitter_0.ply"
# )
mesh_nf = trimesh.load_mesh(
    "/home/pauldelseith/HyperDiffusion_semantic_decomposition/data/chair_100000_pc_occ_in_out_True/chair_arm_filled.obj.npy"
)

pc = np.load(pc_path / "chair_arm_filled.obj.npy")
pc_points = pc[pc[:, -1] == 1, :3]
print(pc_points.max())
print(pc_points.min())
