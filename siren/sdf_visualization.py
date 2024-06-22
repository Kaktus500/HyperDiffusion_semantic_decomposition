import sys
from pathlib import Path

import click
from hydra import compose, initialize
from omegaconf import DictConfig

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)  # TODO: Fix this for debug ...
from progressbar import ProgressBar

from siren import sdf_meshing
from siren.experiment_scripts.test_sdf import SDFDecoder


def generate_mesh_from_sdf(
    checkpoint_path: Path, output_file_path: Path, cfg: DictConfig, *,
    split_sdf: bool = False
) -> None:
    sdf_decoder = SDFDecoder(
        cfg.model_type,
        str(checkpoint_path),
        "nerf" if cfg.model_type == "nerf" else "mlp",
        cfg,
    )
    if output_file_path.parent.is_dir() is False:
        raise ValueError(f"Output directory {output_file_path.parent} does not exist.")
    if split_sdf:
        split_sdf_create_mesh(sdf_decoder, output_file_path, cfg)
    else:
        # multiply sdf by -1 to get the correct normal direction or do direction in lewiner
        sdf_meshing.create_mesh(
            sdf_decoder,
            str(output_file_path),
            N=256,
            level=0 if cfg.output_type == "occ" and cfg.out_act == "sigmoid" else 0,
        )

def split_sdf_create_mesh(
        sdf_decoder: SDFDecoder, output_file_path: Path, cfg: DictConfig
) -> None:
    for j in range(3):
        sdf_meshing.create_mesh(sdf_decoder,
            str(output_file_path) + f"_part_{j}.obj",
            N=256,
            level=0
            if cfg.output_type == "occ" and cfg.out_act == "sigmoid"
            else 0,
            freeze = True if j < 2 else False,
            part = j
        )


@click.group()
def cli():
    """CLI group for mesh generation from SDF."""
    pass


@cli.command()
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
@click.option(
    "--split_sdf",
    is_flag=True,
    help="Part wise mesh generation for a split sdf.",
)
def cli_generate_mesh_from_sdf(
    checkpoint_path: Path, output_file_path: Path, config_name: str, *, split_sdf: bool = False
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
    generate_mesh_from_sdf(checkpoint_path, output_file_path, cfg, split_sdf=split_sdf)


@cli.command()
@click.argument(
    "checkpoint_folder_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.argument(
    "output_folder_path",
    type=click.Path(path_type=Path),
)
@click.argument(
    "config_name",
    type=str,
)
@click.option(
    "--filter_string",
    type=str,
    default="*",
    help="Filter string to select specific checkpoint files.",
)
@click.option(
    "--split_sdf",
    is_flag=True,
    help="Part wise mesh generation for a split sdf.",
)
def cli_generate_meshes_from_sdf(
    checkpoint_folder_path: Path,
    output_folder_path: Path,
    config_name: str,
    filter_string: str,
    *,
    split_sdf: bool = False
) -> None:
    """CLI function to generate meshes for multiple trained SDF models.

    Args:
        checkpoint_path: Path to the checkpoint file.
        output_file_path: Path to the output file.
        config_name: Configuration parameters for the mesh generation.
        filter_string: Filter string to select specific checkpoint files.
    """
    config_path_relative = Path("../configs/overfitting_configs")
    with initialize(version_base=None, config_path=str(config_path_relative)):
        cfg = compose(config_name=config_name)
    progress_bar = ProgressBar(
        maxval=len(list(checkpoint_folder_path.glob(filter_string)))
    ).start()
    for checkpoint_path in checkpoint_folder_path.glob(filter_string):
        output_file_path = output_folder_path / f"{checkpoint_path.stem}"
        generate_mesh_from_sdf(checkpoint_path, output_file_path, cfg, split_sdf=split_sdf)
        progress_bar.update(progress_bar.currval + 1)
    progress_bar.finish()


if __name__ == "__main__":
    cli()
