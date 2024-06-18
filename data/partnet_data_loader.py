"""
Load PartNet data and parse it into semantic labels, based on PartNet repo.
"""

import sys
from pathlib import Path
from typing import Dict, List, Union

import click

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)  # TODO: Fix this for debug ...
import copy
import json

import h5py
import numpy as np
from progressbar import ProgressBar

from helpers import HYPER_DIFF_DIR


def load_file(fn):
    with open(fn, "r") as fin:
        lines = [line.rstrip().split() for line in fin.readlines()]
        pts = np.array(
            [[float(line[0]), float(line[1]), float(line[2])] for line in lines],
            dtype=np.float32,
        )
        nor = np.array(
            [[float(line[3]), float(line[4]), float(line[5])] for line in lines],
            dtype=np.float32,
        )
        rgb = np.array(
            [[int(line[6]), int(line[7]), int(line[8])] for line in lines],
            dtype=np.float32,
        )
        opacity = np.array([float(line[9]) for line in lines], dtype=np.float32)
        return pts, nor, rgb, opacity


def load_label(fn):
    with open(fn, "r") as fin:
        label = np.array(
            [int(item.rstrip()) for item in fin.readlines()], dtype=np.int32
        )
        return label


def save_h5(fn, pts, nor, rgb, opacity, label):
    fout = h5py.File(fn, "w")
    fout.create_dataset(
        "pts", data=pts, compression="gzip", compression_opts=4, dtype="float32"
    )
    fout.create_dataset(
        "nor", data=nor, compression="gzip", compression_opts=4, dtype="float32"
    )
    fout.create_dataset(
        "rgb", data=rgb, compression="gzip", compression_opts=4, dtype="uint8"
    )
    fout.create_dataset(
        "opacity", data=opacity, compression="gzip", compression_opts=4, dtype="float32"
    )
    fout.create_dataset(
        "label", data=label, compression="gzip", compression_opts=4, dtype="int32"
    )
    fout.close()


def save_json(fn, data):
    with open(fn, "w") as fout:
        json.dump(data, fout)


def get_all_leaf_ids(record):
    if "children" in record.keys():
        out = []
        for item in record["children"]:
            out += get_all_leaf_ids(item)
        return out
    elif "objs" in record.keys():
        return [record["id"]]
    else:
        print("ERROR: no children key nor objs key!")
        exit(1)


def get_all_leaf_objs(record):
    if "children" in record.keys():
        out = []
        for item in record["children"]:
            out += get_all_leaf_objs(item)
        return out
    elif "objs" in record.keys():
        return record["objs"]
    else:
        print("ERROR: no children key nor objs key!")
        exit(1)


def traverse(record, cur_name, new_result: List, node_mapping: Dict):
    if len(cur_name) == 0:
        cur_name = record["name"]
    else:
        cur_name = cur_name + "/" + record["name"]
    if cur_name in node_mapping.keys():
        new_part_name = node_mapping[cur_name]
        leaf_id_list = get_all_leaf_ids(record)
        leaf_obj_list = get_all_leaf_objs(record)
        new_result.append(
            {
                "leaf_id_list": leaf_id_list,
                "part_name": new_part_name,
                "leaf_obj_list": leaf_obj_list,
            }
        )
    if "children" in record.keys():
        for item in record["children"]:
            traverse(item, cur_name, new_result, node_mapping)


def normalize_pc(pts):
    x_max = np.max(pts[:, 0])
    x_min = np.min(pts[:, 0])
    x_mean = (x_max + x_min) / 2
    y_max = np.max(pts[:, 1])
    y_min = np.min(pts[:, 1])
    y_mean = (y_max + y_min) / 2
    z_max = np.max(pts[:, 2])
    z_min = np.min(pts[:, 2])
    z_mean = (z_max + z_min) / 2
    pts[:, 0] -= x_mean
    pts[:, 1] -= y_mean
    pts[:, 2] -= z_mean
    scale = np.sqrt(np.max(np.sum(pts**2, axis=1)))
    pts /= scale
    return pts


def extract_ins_seg_annotations(
    data_folder_path: Path, category: str, split: str, n_shapes: Union[int, None] = None
) -> None:
    """Extract instance segementation annotations from the given category and split of the PartNet dataset.

    Args:
        category (str): Category of the PartNet dataset.
        split (str): Split of the PartNet dataset has to be one of train, val, test.
        n_shapes (int | None): Number of shapes per .json file. If None, all shapes are extracted into one file.
    """
    in_fn = f"stats/train_val_test_split/{category}.{split}.json"
    in_fn = HYPER_DIFF_DIR / "data" / "partnet" / in_fn
    with open(in_fn, "r") as fin:
        item_list = json.load(fin)
    in_fn = f"stats/merging_hierarchy_mapping/{category}.txt"
    in_fn = HYPER_DIFF_DIR / "data" / "partnet" / in_fn
    with open(in_fn, "r") as fin:
        node_mapping = {
            d.rstrip().split()[0]: d.rstrip().split()[1] for d in fin.readlines()
        }

    batch_record = []
    out_dir = None

    t = 0
    k = 0
    progress_bar = ProgressBar(maxval=len(item_list)).start()
    for item_id, item in enumerate((item_list)):
        in_res_fn = f"{item['anno_id']}/result.json"
        in_res_fn = data_folder_path / in_res_fn
        if not in_res_fn.exists():
            print(f"File {in_res_fn} does not exist, skipping!")
            continue

        with open(in_res_fn, "r") as fin:
            data = json.load(fin)
        new_result = []
        traverse(data[0], "", new_result, node_mapping)

        new_record = copy.deepcopy(item)
        new_record["ins_seg"] = new_result
        batch_record.append(new_record)
        progress_bar.update(item_id + 1)
        k += 1

        # once finished with all samples from input file or specified number of samples reached, store results
        if k == n_shapes or item_id + 1 == len(item_list):
            out_dir = f"ins_seg/{category}"
            out_dir = HYPER_DIFF_DIR / "data" / "partnet" / out_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            out_fn_prefix = out_dir / f"{split}-{t:02d}"
            save_json(str(out_fn_prefix) + ".json", batch_record)
            t += 1
            k = 0
            batch_record = []
    progress_bar.finish()
    print("Instance segmentation annotations extracted.")
    print(f"Data stored in: {out_dir}")


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
    help="Number of shapes per .json file. If None, all shapes are extracted into one file.",
)
def cli_extract_ins_seg_annotations(
    data_folder_path: Path, category: str, split: str, n_shapes: Union[int, None] = None
) -> None:
    """Extract instance segementation annotations from the given category and split of the PartNet dataset.

    Args:
        category (str): Category of the PartNet dataset.
        split (str): Split of the PartNet dataset has to be one of train, val, test.
        n_shapes (int | None): Number of shapes per .json file. If None, all shapes are extracted into one file.
    """
    extract_ins_seg_annotations(data_folder_path, category, split, n_shapes)


if __name__ == "__main__":
    # data_folder_path = Path("/home/pauldelseith/dataset_storage/partnet") / "data_v0"
    cli_extract_ins_seg_annotations()