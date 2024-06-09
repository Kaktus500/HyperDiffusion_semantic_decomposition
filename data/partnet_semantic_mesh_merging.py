import sys
from pathlib import Path
from typing import Dict, List

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)  # TODO: Fix this for debug ...
import json

import pymeshlab
from pymeshlab import Percentage
import trimesh

from helpers import HYPER_DIFF_DIR

category = "Chair_small"
split = "train"
index = "00"
annotation_dir = (
    HYPER_DIFF_DIR
    / "data"
    / "partnet"
    / "ins_seg_h5"
    / category
    / f"{split}-{index}-sem_seg_ids.json"
)

mesh_dir = HYPER_DIFF_DIR / "data" / "chair"

with open(annotation_dir, "r") as f:
    shape_segementation_id_mapping: Dict[str, Dict[str, List]] = json.load(f)

sem_seg_items = {}
for item_id, item in shape_segementation_id_mapping.items():
    print(item_id)
    sem_seg_item = {}
    for part_name, part in item.items():
        merged_mesh = trimesh.Trimesh()
        for leaf_id, leaf_obj in part:
            print(leaf_id, leaf_obj)
            mesh_file = mesh_dir / item_id / "objs" / f"{leaf_obj}.obj"
            mesh = trimesh.load_mesh(mesh_file)
            merged_mesh += mesh
        merged_mesh.export(
            file_type="obj", file_obj=mesh_dir / item_id / f"{part_name}.obj"
        )
        # try hole filling
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(mesh_dir / item_id / f"{part_name}.obj"))
        no_vertextes = ms.current_mesh().vertex_number()
        ms.meshing_merge_close_vertices(threshold=Percentage(5))
        ms.meshing_remove_duplicate_faces()
        ms.meshing_repair_non_manifold_edges(method="Split Vertices")
        result = ms.meshing_close_holes(
            maxholesize=no_vertextes // 20, selfintersection=False
        )
        iteration = 0
        while result["new_faces"] > 0 and iteration < 15:
            result = ms.meshing_close_holes(
                maxholesize=no_vertextes // 20, selfintersection=False
            )
            iteration += 1
        print("Iteration:", iteration)
        ms.save_current_mesh(str(mesh_dir / item_id / f"{part_name}_filled.obj"))
        sem_seg_item[part_name] = merged_mesh
    sem_seg_items[item_id] = sem_seg_item
