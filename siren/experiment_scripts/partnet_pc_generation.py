import sys
from pathlib import Path
from typing import Any, Dict, List, Union, Tuple

sys.path.append(
    str(Path(__file__).resolve().parent.parent.parent)
)  # TODO: Fix this for debug ...
import subprocess

import igl
import numpy as np
import trimesh
from progressbar import ProgressBar

from helpers import HYPER_DIFF_DIR


def sample_occupancy_grid_from_mesh(
    mesh: trimesh.Trimesh, n_points_uniform: int, n_points_surface: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample points from 3D space to approximate an occupancy grid.

    Args:
        mesh: The mesh to sample points from.
        n_points_uniform: The number of points to sample uniformly from 3D space.
        n_points_surface: The number of points to sample from the surface of the mesh.
    """
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
    return points, occupancies


def generate_shape_occupancy_pc(
    mesh_parts: Dict[str, Tuple[trimesh.Trimesh, Path]],
    cfg: Dict[str, Any], *, check_occupancy: bool = True
) -> Union[Dict[str, np.ndarray], None]:
    """Generate a normalized point cloud for the parts of a shape.

    Args:
        mesh_parts: A dictionary mapping part names to their corresponding meshes.
        cfg: A dictionary containing the configuration parameters for the point cloud generation.
    """

    total_points = cfg["n_points"]
    n_points_uniform = total_points
    n_points_surface = total_points

    part_point_clouds: Dict[str, np.ndarray] = {}
    for part_name, mesh in mesh_parts.items():
        points, occupancies = sample_occupancy_grid_from_mesh(
            mesh[0], n_points_uniform, n_points_surface
        )
        # check whether a reasonable number of points inside the shape was found
        if check_occupancy and occupancies.sum() < 0.1 * total_points:
            # try fixing the shape using ManifoldPlus
            command = f"~/ManifoldPlus/build/manifold --input {mesh[1]} --output {mesh[1]} --depth 8"
            try:
                subprocess.run(
                    command,
                    shell=True,
                    text=True,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except subprocess.CalledProcessError:
                return None
            manifold_mesh = trimesh.load(mesh[1])
            points, occupancies = sample_occupancy_grid_from_mesh(
                manifold_mesh, n_points_uniform, n_points_surface
            )
        # if there is still a small amount of points, discard the shape
        if check_occupancy and occupancies.sum() < 0.05 * total_points:
            return None
        point_cloud = points
        point_cloud = np.hstack((point_cloud, occupancies))
        coords = point_cloud[:, :3]
        normals = point_cloud[:, 3:]
        point_cloud_xyz = np.hstack((coords, normals))
        part_point_clouds[part_name] = point_cloud_xyz
    return part_point_clouds

def generate_shape_pc(mesh: trimesh.Trimesh, n_points: int) -> np.ndarray:
    """Generate a normalized point cloud for a shape.

    Args:
        mesh: The mesh of the shape.
        n_points: The number of points to sample.
    """
    points = mesh.sample(n_points)
    return points

def generate_shapes_pcs(category: str, parts: List[str], cfg: Dict[str, Any], include_part_names: Union[List[str], None] = None) -> None:
    """Generate point clouds for points on the surface of the shapes."""
    pc_out_dir = (
        HYPER_DIFF_DIR
        / "data"
        / "partnet"
        / "sem_seg_meshes"
        / f"{category}_{cfg['n_points']}_pc"
    )
    pc_out_dir.mkdir(parents=True, exist_ok=True)
    # extract unique shape ids from parts list
    shape_ids = {part.split("_")[0] for part in parts}
    shape_ids = list(shape_ids)
    progress_bar = ProgressBar(maxval=len(shape_ids)).start()
    mesh_parts_dir = HYPER_DIFF_DIR / "data" / "partnet" / "sem_seg_meshes" / category
    # iterate over shapes and generate normalized point clouds
    for shape_id in shape_ids:
        shape_mesh = trimesh.Trimesh()
        for mesh_path in mesh_parts_dir.glob(f"{shape_id}_*.obj"):
            mesh = trimesh.load_mesh(mesh_path)
            part_name = mesh_path.stem.split("_")[-1]
            if include_part_names is not None and part_name in include_part_names:
                shape_mesh += mesh
        mesh_pc = generate_shape_pc(shape_mesh, cfg["n_points"])
        pc_path = pc_out_dir / f"{shape_id}.npy"
        np.save(pc_path, mesh_pc)
        progress_bar.update(progress_bar.currval + 1)
    progress_bar.finish()
    print(f"Generated point clouds for {len(shape_ids)} shapes")
    print(f"Point clouds saved at {pc_out_dir}")


def generate_shapes_occupancy_pcs(
    category: str, parts: List[str], cfg: Dict[str, Any], *, check_occupancy: bool=True
) -> List[str]:
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
    shapes_skipped = 0
    good_mesh_ids: List[str] = []
    # iterate over shapes and generate normalized point clouds
    for shape_id in shape_ids:
        mesh_parts: Dict[str, trimesh.Trimesh] = {}
        for mesh_path in mesh_parts_dir.glob(f"{shape_id}_*.obj"):
            mesh = trimesh.load_mesh(mesh_path)
            mesh_name = mesh_path.name
            mesh_parts[mesh_name] = (mesh, mesh_path)
        normalized_mesh_parts = generate_shape_occupancy_pc(mesh_parts, cfg, check_occupancy=check_occupancy)
        if normalized_mesh_parts is None:
            progress_bar.update(progress_bar.currval + 1)
            shapes_skipped += 1
            continue
        good_mesh_ids.append(shape_id)
        for part_name, pc in normalized_mesh_parts.items():
            colors = np.zeros((pc.shape[0], 4), dtype=np.uint8)
            colors[:, 3] = 1
            colors[pc[:, 3] == 1, 0] = 255
            colors[pc[:, 3] == 0, 1] = 255
            trimesh.points.PointCloud(pc[:, :3], colors=colors).export(
                pc_out_dir / f"{part_name}_pc.obj"
            )
            pc_path = pc_out_dir / f"{part_name}.npy"
            np.save(pc_path, pc)
        progress_bar.update(progress_bar.currval + 1)
    progress_bar.finish()
    print(f"Skipped {shapes_skipped} shapes")
    print(f"Generated point clouds for {len(shape_ids) - shapes_skipped} shapes")
    print(f"Point clouds saved at {pc_out_dir}")
    with open(pc_out_dir / "good_mesh_ids.lst", "w") as f:
        for mesh_id in good_mesh_ids:
            f.write(f"{mesh_id}\n")
    return good_mesh_ids


if __name__ == "__main__":
    category = "Chair_reduced_full"
    split = "test"
    dataset_dir = HYPER_DIFF_DIR / "data" / "partnet" / "sem_seg_meshes" / category
    file_names = np.genfromtxt(
        dataset_dir / f"{split}_split.lst",
        dtype="str",
    )
    # parts = [
    #     file.name
    #     for file in dataset_dir.iterdir()
    #     if file.name not in {"train_split.lst", "val_split.lst", "test_split.lst"}
    #     and file.name in file_names
    # ]
    parts = [
        file.name
        for file in dataset_dir.iterdir()
        if file.name not in {"train_split.lst", "val_split.lst", "test_split.lst"}
        and file.stem.split("_")[0] in file_names
    ]
    cfg = {
        "save_pc": True,
        "in_out": True,
        "mlp_config": {"move": False},
        "strategy": "save_pc",
        "shape_modify": "no",
        "n_points": 2048,
        "output_type": "occ",
    }
    # generate_shapes_occupancy_pcs(category, parts, cfg, check_occupancy=False)
    generate_shapes_pcs(category, parts, cfg, ["base", "seat","arm","back"])
