import sys
from pathlib import Path
from typing import Dict, List, Tuple, Union

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)  # TODO: Fix this for debug ...
import json
import shutil

import click
import pymeshlab
import trimesh
from progressbar import ProgressBar
from pymeshlab import Percentage
from pymeshlab.pmeshlab import PyMeshLabException

from helpers import HYPER_DIFF_DIR


def sem_seg_mesh_merging(
    data_folder_path: Path, category: str, split: str, n_shapes: Union[int, None] = None
) -> None:
    """Merge meshes of parts that belong to the same semantic category according to the semantic segementation annotations."""
    annotation_dir = HYPER_DIFF_DIR / "data" / "partnet" / "sem_seg" / category

    mesh_dir = data_folder_path

    mesh_output_dir = HYPER_DIFF_DIR / "data" / "partnet" / "sem_seg_meshes" / category

    print("Merging meshes ...")

    merged_meshes = 0
    meshes_successfully_merged = 0
    ms = pymeshlab.MeshSet()
    for annotation_file_path in annotation_dir.glob(f"{split}-*sem_seg_ids.json"):
        with open(annotation_file_path, "r") as f:
            shape_segementation_id_mapping: Dict[
                str, Dict[str, List[Tuple[int, List[str]]]]
            ] = json.load(f)
        if n_shapes is None:
            progress_bar = ProgressBar(
                maxval=len(shape_segementation_id_mapping)
            ).start()
        else:
            progress_bar = ProgressBar(
                maxval=min(len(shape_segementation_id_mapping), n_shapes)
            ).start()
        loop_progress = 0
        merge_failed = False
        for item_id, item in shape_segementation_id_mapping.items():
            part_output_dir = mesh_output_dir / item_id
            part_output_dir.mkdir(parents=True, exist_ok=True)
            for part_name, part in item.items():
                merged_mesh = trimesh.Trimesh()
                for leaf_id, leaf_objs in part:
                    for leaf_obj in leaf_objs:
                        mesh_file = mesh_dir / item_id / "objs" / f"{leaf_obj}.obj"
                        mesh = trimesh.load_mesh(mesh_file)
                        merged_mesh += mesh
                merged_mesh.export(
                    file_type="obj", file_obj=mesh_dir / item_id / f"{part_name}.obj"
                )
                # try hole filling
                ms.load_new_mesh(str(mesh_dir / item_id / f"{part_name}.obj"))
                no_vertexes = ms.current_mesh().vertex_number()
                ms.meshing_merge_close_vertices(threshold=Percentage(5))
                ms.meshing_remove_duplicate_faces()
                ms.meshing_repair_non_manifold_edges(method="Split Vertices")
                try:
                    result = ms.meshing_close_holes(
                        maxholesize=no_vertexes // 20, selfintersection=False
                    )
                    iteration = 0
                    while result["new_faces"] > 0 and iteration < 15:
                        result = ms.meshing_close_holes(
                            maxholesize=no_vertexes // 20, selfintersection=False
                        )
                        iteration += 1
                    ms.save_current_mesh(str(part_output_dir / f"{part_name}.obj"))
                except PyMeshLabException as e:
                    if part_output_dir.exists() and part_output_dir.is_dir():
                        shutil.rmtree(part_output_dir)
                    merge_failed = True
                    break
            ms.clear()
            loop_progress += 1
            merged_meshes += 1
            if not merge_failed:
                meshes_successfully_merged += 1
            merge_failed = False
            progress_bar.update(loop_progress)
            if merged_meshes == n_shapes:
                break
        progress_bar.finish()
        if merged_meshes == n_shapes:
            break
    print(f"Merged {meshes_successfully_merged} meshes, stored in {mesh_output_dir}")


@click.command()
@click.argument("data_folder_path", type=Path)
@click.argument("category", type=str)
@click.argument(
    "split",
    type=str,
)
@click.option(
    "--n_shapes",
    type=int,
    default=None,
    help="Number of shapes to process. If not specified, all shapes of the given split are processed.",
)
def cli_sem_seg_mesh_merging(
    data_folder_path: Path, category: str, split: str, n_shapes: Union[int, None] = None
) -> None:
    sem_seg_mesh_merging(data_folder_path, category, split, n_shapes)


if __name__ == "__main__":
    data_folder_path = Path("/home/pauldelseith/dataset_storage/partnet") / "data_v0"
    # sem_seg_mesh_merging(data_folder_path, "Chair", "faketrain", 10)
    cli_sem_seg_mesh_merging()
