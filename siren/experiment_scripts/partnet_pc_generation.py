import sys
from pathlib import Path
from typing import Dict, List, Any

sys.path.append(
    str(Path(__file__).resolve().parent.parent.parent)
)  # TODO: Fix this for debug ...
from siren import dataio
from helpers import HYPER_DIFF_DIR
import trimesh
import numpy as np
import igl


def generate_normalized_shape_pc(
    mesh_parts: Dict[str, trimesh.Trimesh],
    cfg: Dict[str, Any],
) -> Dict[str, np.ndarray]:
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

    total_points = cfg["n_points"]  # 100000
    n_points_uniform = total_points  # int(total_points * 0.5)
    n_points_surface = total_points  # total_points

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
        print(points.shape, occupancies.shape, occupancies.sum())
        point_cloud = points
        point_cloud = np.hstack((point_cloud, occupancies))
        print(point_cloud.shape, points.shape, occupancies.shape)
        coords = point_cloud[:, :3]
        normals = point_cloud[:, 3:]
        point_cloud_xyz = np.hstack((coords, normals))
        part_point_clouds[part_name] = point_cloud_xyz
    return part_point_clouds


if __name__ == "__main__":
    part_id = "35508"
    shape_name = "chair"
    dataset_dir = HYPER_DIFF_DIR / "data" / shape_name
    mesh_parts_dir = dataset_dir / part_id / "sem_seg_parts"
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
    }
    mesh_parts: Dict[str, trimesh.Trimesh] = {}
    for mesh_path in mesh_parts_dir.iterdir():
        mesh = trimesh.load_mesh(mesh_path)
        mesh_name = mesh_path.name
        mesh_parts[mesh_name] = mesh
    normalized_mesh_parts = generate_normalized_shape_pc(mesh_parts, cfg)
    pc_out_dir = (
        dataset_dir.parent
        / f"{shape_name}_{n_points}_pc_{output_type}_in_out_{cfg['in_out']}"
    )
    pc_out_dir.mkdir(parents=True, exist_ok=True)
    for part_name, pc in normalized_mesh_parts.items():
        pc_path = pc_out_dir / f"{part_name}.npy"
        np.save(pc_path, pc)
        print(f"Saved point cloud for part {part_name} to {pc_path}")
