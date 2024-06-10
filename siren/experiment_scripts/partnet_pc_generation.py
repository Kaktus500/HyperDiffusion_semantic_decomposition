import sys
from pathlib import Path
from typing import Any, Dict, List, Union

sys.path.append(
    str(Path(__file__).resolve().parent.parent.parent)
)  # TODO: Fix this for debug ...
import igl
import numpy as np
import trimesh
from progressbar import ProgressBar

from helpers import HYPER_DIFF_DIR


def generate_normalized_shape_pc(
    mesh_parts: Dict[str, trimesh.Trimesh],
    cfg: Dict[str, Any],
) -> Union[Dict[str, np.ndarray], None]:
    """Generate a normalized point cloud for the parts of a shape."""

    combined_mesh = trimesh.Trimesh()
    for mesh in mesh_parts.values():
        combined_mesh += mesh

    # compute normalization metrics
    vertices = combined_mesh.vertices
    vertices_mean = np.mean(vertices, axis=0, keepdims=True)
    vertices -= vertices_mean
    vertices_max = np.amax(vertices)
    vertices_min = np.amin(vertices)
    vertices_scaling = 0.5 * 0.95 / (max(abs(vertices_min), abs(vertices_max)))

    # normalize meshes
    for part_name, mesh in mesh_parts.items():
        mesh.vertices -= vertices_mean
        mesh.vertices *= vertices_scaling

    total_points = cfg["n_points"]
    n_points_uniform = total_points
    n_points_surface = total_points

    part_point_clouds: Dict[str, np.ndarray] = {}
    for part_name, mesh in mesh_parts.items():
        points_uniform = np.random.uniform(-0.5, 0.5, size=(n_points_uniform, 3))
        points_surface = mesh.sample(n_points_surface)
        points_surface += 0.01 * np.random.randn(n_points_surface, 3)
        points = np.concatenate([points_surface, points_uniform], axis=0)
        inside_surface_values = igl.fast_winding_number_for_meshes(
            mesh.vertices, mesh.faces, points
        )
        thresh = 0.5
        occupancies_winding = np.piecewise(
            inside_surface_values,
            [inside_surface_values < thresh, inside_surface_values >= thresh],
            [0, 1],
        )
        occupancies = occupancies_winding[..., None]
        if occupancies.sum() < 1000:
            print("Not enough points inside the mesh, there is likely a problem.")
            return None
        point_cloud = points
        point_cloud = np.hstack((point_cloud, occupancies))
        coords = point_cloud[:, :3]
        normals = point_cloud[:, 3:]
        point_cloud_xyz = np.hstack((coords, normals))
        part_point_clouds[part_name] = point_cloud_xyz
    return part_point_clouds


def generate_shapes_pcs(category: str, parts: List[str], cfg: Dict[str, Any]) -> None:
    """Geneerate point clouds for list of given shapes.

    Applies shape level normalization before point cloud extraction.
    """
    pc_out_dir = (
        HYPER_DIFF_DIR
        / "data"
        / "partnet"
        / "sem_seg_meshes"
        / f"{category}_{cfg['n_points']}_pc_{cfg['output_type']}_in_out_{cfg['in_out']}"
    )
    pc_out_dir.mkdir(parents=True, exist_ok=True)
    # extract unique shape ids from parts list
    shape_ids = set([part.split("_")[0] for part in parts])
    shape_ids = list(shape_ids)
    progress_bar = ProgressBar(maxval=len(shape_ids)).start()
    mesh_parts_dir = HYPER_DIFF_DIR / "data" / "partnet" / "sem_seg_meshes" / category
    for shape_id in shape_ids:
        mesh_parts: Dict[str, trimesh.Trimesh] = {}
        for mesh_path in mesh_parts_dir.glob(f"{shape_id}_*.obj"):
            mesh = trimesh.load_mesh(mesh_path)
            mesh_name = mesh_path.name
            mesh_parts[mesh_name] = mesh
        normalized_mesh_parts = generate_normalized_shape_pc(mesh_parts, cfg)
        if normalized_mesh_parts is None:
            print(f"Skipping shape {shape_id} due to insufficient points.")
            continue
        for part_name, pc in normalized_mesh_parts.items():
            pc_path = pc_out_dir / f"{part_name}.npy"
            np.save(pc_path, pc)
        progress_bar.update(progress_bar.currval + 1)
    progress_bar.finish()


if __name__ == "__main__":
    category = "Chair"
    split = "train"
    dataset_dir = HYPER_DIFF_DIR / "data" / "partnet" / "sem_seg_meshes" / category
    file_names = np.genfromtxt(dataset_dir / f"{split}_split.lst", dtype="str")
    parts = [
        file.name
        for file in dataset_dir.iterdir()
        if file.name not in {"train_split.lst", "val_split.lst", "test_split.lst"}
        and file.name in file_names
    ]
    shape_name = "Chair"
    on_surface_points = 2048
    output_type = "occ"
    n_points = 100000
    cfg = {
        "save_pc": True,
        "in_out": True,
        "mlp_config": {"move": False},
        "strategy": "save_pc",
        "shape_modify": "no",
        "n_points": 100000,
        "output_type": "occ",
    }
    generate_shapes_pcs(category, parts, cfg)
