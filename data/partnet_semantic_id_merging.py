import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)  # TODO: Fix this for debug ...
import json
from collections import defaultdict

import click
from progressbar import ProgressBar

from helpers import HYPER_DIFF_DIR


def merge_ins_seg_categories(category: str, split: str) -> None:
    """Merge same instances of instance segementations to get semantic segmentations.
    E.g. for a chair, merge all instances of the chair legs to get the chair base.

    Args:
        category (str): The category of the shapes.
        split (str): The split of the PartNet dataset, has to be one of train, val, test.
    """
    ins_seg_dir = HYPER_DIFF_DIR / "data" / "partnet" / "ins_seg" / category
    sem_seg_dir = HYPER_DIFF_DIR / "data" / "partnet" / "sem_seg" / category
    for annotation_file_path in ins_seg_dir.iterdir():
        if str(annotation_file_path.name).split("-")[0] != split:
            continue
        index = str(annotation_file_path.name).split("-")[1].split(".")[0]
        with open(annotation_file_path, "r") as fin:
            annotations = json.load(fin)
        print(f"Processing {category} {split} {index}")
        progress_bar = ProgressBar(maxval=len(annotations)).start()
        shape_segementation_id_mapping = {}
        for i, item in enumerate(annotations):
            segmentation = item["ins_seg"]
            second_level_segmentation_ids = defaultdict(set)
            for part in segmentation:
                part_name = part["part_name"]
                part_components = part_name.split("/")
                # only add leaf nodes
                if (
                    len(part_components) < 2
                    or len(part["leaf_id_list"]) > 1
                    or len(part["leaf_id_list"]) == 0
                ):
                    continue
                segmentation_label = part_components[1]
                second_level_segmentation_ids[segmentation_label].add(
                    (part["leaf_id_list"][0], part["leaf_obj_list"][0])
                )
            if second_level_segmentation_ids:
                for key, value in second_level_segmentation_ids.items():
                    second_level_segmentation_ids[key] = list(value)
                shape_segementation_id_mapping[item["anno_id"]] = (
                    second_level_segmentation_ids
                )
            progress_bar.update(i + 1)
        progress_bar.finish()
        sem_seg_dir.mkdir(parents=True, exist_ok=True)
        with open(
            sem_seg_dir / f"{split}-{index}-sem_seg_ids.json",
            "w",
        ) as fout:
            json.dump(shape_segementation_id_mapping, fout)
    print(f"Instance segmentation annotations merged for {category} {split}")
    print(f"Data stored in: {sem_seg_dir}")


@click.command()
@click.argument("category", type=str)
@click.argument("split", type=str)
def cli_merge_ins_seg_categories(category: str, split: str) -> None:
    """CLI command to merge same instances of instance segementations to get semantic segmentations.
    E.g. for a chair, merge all instances of the chair legs to get the chair base.

    Args:
        category (str): The category of the shapes.
        split (str): The split of the PartNet dataset, has to be one of train, val, test.
    """
    merge_ins_seg_categories(category, split)


if __name__ == "__main__":
    cli_merge_ins_seg_categories()
