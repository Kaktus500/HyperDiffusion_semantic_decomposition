"""Reproduces Sec. 4.2 in main paper and Sec. 4 in Supplement.
"""
import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent.parent)
)  # TODO: Fix this for debug ...
import copy
import os

# Enable import from parent package
from functools import partial

import hydra
import torch
import trimesh
import wandb
from omegaconf import DictConfig, open_dict

from helpers import HYPER_DIFF_DIR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader

from hd_utils import render_mesh
from mlp_models import MLP3D
from mlp_models_split import MLP3D_Split
from siren import (
    dataio,
    dataio_partnet,
    loss_functions,
    sdf_meshing,
    training_partnet_split,
    utils,
)
from siren.experiment_scripts.test_sdf import SDFDecoder


def get_model(cfg):
    if cfg.model_type == "mlp_3d":
        model = MLP3D(**cfg.mlp_config)
    nparameters = sum(p.numel() for p in model.parameters())
    print(model)
    print("Total number of parameters: %d" % nparameters)

    return model


@hydra.main(
    version_base=None,
    config_path="../../configs/overfitting_configs",
    config_name="overfit_chair",
)
def main(cfg: DictConfig):
    wandb.init(
        project="hyperdiffusion_overfitting",
        dir=cfg.wandb_dir,
        config=dict(cfg),
        mode="online",
    )
    first_state_dict = None
    if cfg.strategy == "same_init":
        first_state_dict = get_model(cfg).state_dict()
    x_0s = []
    with open_dict(cfg):
        cfg.mlp_config.output_type = cfg.output_type
    curr_lr = cfg.lr
    root_path = os.path.join(str(HYPER_DIFF_DIR), cfg.logging_root, cfg.exp_name)
    print(f"Root path {root_path}")
    multip_cfg = cfg.multi_process
    dataset_folder = str(HYPER_DIFF_DIR / Path(cfg.dataset_folder))
    files = os.listdir(dataset_folder)


    ############################## Here we load an object for each sample that contains a list of class labels ##############################
    # e.g. files_labeled = {"chair1": ["base", "back"], "chair2": ["base", "back"]}

    files_labeled = {}
    for file in files:
        file = file.split(".")[0]
        name, label = file.split("_")
        if name not in files_labeled:
            files_labeled[name] = []
        files_labeled[name].append(label)
 
    ######################################################################################################################################

    if multip_cfg.enabled:
        if cfg.strategy == "first_weights":
            first_state_dict = torch.load(
                os.path.join(root_path, multip_cfg.first_weights_name)
            )

    for i, file in enumerate(files_labeled.keys()):
        filename = file

        sdf_dataset = dataio_partnet.PointCloud( # The dataset class is modified to include the class labels
            dataset_folder,
            file,
            files_labeled[file],
            on_surface_points=cfg.batch_size,
        )

        dataloader = DataLoader(
            sdf_dataset,
            shuffle=True,
            batch_size=1,
            pin_memory=True,
            num_workers=0,
        )

        # Define the model.
        model = get_model(cfg).cuda()

        # Define the loss
        loss_fn = loss_functions.sdf
        if cfg.output_type == "occ":
            loss_fn = (
                loss_functions.occ_tanh
                if cfg.out_act == "tanh"
                else loss_functions.occ_sigmoid
            )
        loss_fn = partial(loss_fn, cfg=cfg)
        summary_fn = utils.wandb_sdf_summary

        filename = f"{cfg.output_type}_{filename}"
        checkpoint_path = os.path.join(root_path, f"{filename}_model_final.pth")
        if os.path.exists(checkpoint_path):
            print("Checkpoint exists:", checkpoint_path)
            continue
        if cfg.strategy == "continue":
            if not os.path.exists(checkpoint_path):
                continue
            model.load_state_dict(torch.load(checkpoint_path))
        elif (
            first_state_dict is not None
            and cfg.strategy != "random"
            and cfg.strategy != "first_weights_kl"
        ):
            model.load_state_dict(first_state_dict)

        training_partnet_split.train(
            model=model,
            train_dataloader=dataloader,
            epochs=cfg.epochs,
            lr=curr_lr,
            steps_til_summary=cfg.steps_til_summary,
            epochs_til_checkpoint=cfg.epochs_til_ckpt,
            model_dir=root_path,
            loss_fn=loss_fn,
            summary_fn=summary_fn,
            double_precision=False,
            clip_grad=cfg.clip_grad,
            wandb=wandb,
            filename=filename,
            cfg=cfg,
        )
        if (
            i == 0
            and first_state_dict is None
            and (
                cfg.strategy == "first_weights"
                or cfg.strategy == "first_weights_kl"
            )
            and not multip_cfg.enabled
        ):
            first_state_dict = model.state_dict()
            print(curr_lr)
        state_dict = model.state_dict()

        # Calculate statistics on the MLP
        weights = []
        for weight in state_dict:
            weights.append(state_dict[weight].flatten().cpu())
        weights = torch.hstack(weights)
        x_0s.append(weights)
        tmp = torch.stack(x_0s)
        var = torch.var(tmp, dim=0)
        print(
            var.shape,
            var.mean().item(),
            var.std().item(),
            var.min().item(),
            var.max().item(),
        )
        print(var.shape, torch.var(tmp))

        # For the first 5 data, outputting shapes
        if i < 5:
                sdf_decoder = SDFDecoder(
                    cfg.model_type,
                    checkpoint_path,
                    "nerf" if cfg.model_type == "nerf" else "mlp",
                    cfg,
                )
                os.makedirs(
                    os.path.join(cfg.logging_root, f"{cfg.exp_name}_ply"), exist_ok=True
                )
                if cfg.mlp_config.move:
                    imgs = []
                    for time_val in range(sdf_dataset.total_time):
                        vertices, faces, _ = sdf_meshing.create_mesh(
                            sdf_decoder,
                            os.path.join(
                                cfg.logging_root,
                                f"{cfg.exp_name}_ply",
                                filename + "_" + str(time_val),
                            ),
                            N=256,
                            level=0.5
                            if cfg.output_type == "occ" and cfg.out_act == "sigmoid"
                            else 0,
                            time_val=time_val,
                        )
                        rot_matrix = Rotation.from_euler(
                            "zyx", [45, 180, 90], degrees=True
                        )
                        tmp = copy.deepcopy(faces[:, 1])
                        faces[:, 1] = faces[:, 2]
                        faces[:, 2] = tmp
                        obj = trimesh.Trimesh(rot_matrix.apply(vertices), faces)
                        img, _ = render_mesh(obj)
                        imgs.append(img)
                    imgs = np.array(imgs)
                    imgs = np.transpose(imgs, axes=(0, 3, 1, 2))
                    wandb.log({"animation": wandb.Video(imgs, fps=16)})
                else:
                    for j in range(3):
                        sdf_meshing.create_mesh(
                            sdf_decoder,
                            os.path.join(cfg.logging_root, f"{cfg.exp_name}_ply", f"{filename}_{j}"),
                            N=256,
                            level=0
                            if cfg.output_type == "occ" and cfg.out_act == "sigmoid"
                            else 0,
                            freeze = True if j < 2 else False,
                            part = j
                        )


if __name__ == "__main__":
    main()
