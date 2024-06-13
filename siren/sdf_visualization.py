import sys
from pathlib import Path

import click
from hydra import compose, initialize
from omegaconf import DictConfig

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)  # TODO: Fix this for debug ...
from siren import sdf_meshing
from siren.experiment_scripts.test_sdf import SDFDecoder


def generate_mesh_from_sdf(
    checkpoint_path: Path, output_file_path: Path, cfg: DictConfig
) -> None:
    sdf_decoder = SDFDecoder(
        cfg.model_type,
        str(checkpoint_path),
        "nerf" if cfg.model_type == "nerf" else "mlp",
        cfg,
    )
    if output_file_path.parent.is_dir() is False:
        raise ValueError(f"Output directory {output_file_path.parent} does not exist.")
    sdf_meshing.create_mesh(
        sdf_decoder,
        str(output_file_path),
        N=256,
        level=0 if cfg.output_type == "occ" and cfg.out_act == "sigmoid" else 0,
    )
    # multiply sdf by -1 to get the correct normal direction or do direction in lewiner


@click.command()
@click.argument(
    "checkpoint_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.argument(
    "output_file_path",
    type=click.Path(path_type=Path),
)
@click.argument(
    "config_name",
    type=str,
)
def cli_generate_mesh_from_sdf(
    checkpoint_path: Path, output_file_path: Path, config_name: str
) -> None:
    """CLI function to generate a mesh from a trained SDF model.

    Args:
        checkpoint_path: Path to the checkpoint file.
        output_file_path: Path to the output file.
        cfg: Configuration parameters for the mesh generation.
    """
    config_path_relative = Path("../configs/overfitting_configs")
    with initialize(version_base=None, config_path=str(config_path_relative)):
        cfg = compose(config_name=config_name)
    generate_mesh_from_sdf(checkpoint_path, output_file_path, cfg)


if __name__ == "__main__":
    cli_generate_mesh_from_sdf()
