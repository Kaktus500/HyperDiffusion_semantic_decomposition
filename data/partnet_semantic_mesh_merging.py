import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent)) # TODO: Fix this for debug ...
from helpers import HYPER_DIFF_DIR
import json
from collections import defaultdict
from progressbar import ProgressBar

category = "chair"
split = "train"
index = "00"
annotation_dir = HYPER_DIFF_DIR / "data" / "partnet" / "ins_seg_h5" / category / f"{split}-{index}.json"
with open(annotation_dir, 'r') as fin:
    annotations = json.load(fin)
shape_segementation_id_mapping = {}
progress_bar = ProgressBar(maxval=len(annotations)).start()
for i, item in enumerate(annotations):
    segmentation = item['ins_seg']
    second_level_segmentation_ids = defaultdict(list)
    for part in segmentation:
        part_name = part["part_name"]
        part_components = part_name.split("/")
        # only add leaf nodes
        if len(part_components) < 2 or len(part["leaf_id_list"]) > 1 or len(part["leaf_id_list"]) == 0:
            continue
        segmentation_label = part_components[1]
        second_level_segmentation_ids[segmentation_label].append(part["leaf_id_list"][0])
    if second_level_segmentation_ids:
        shape_segementation_id_mapping[item["anno_id"]] = second_level_segmentation_ids
    progress_bar.update(i+1)
progress_bar.finish()
with open(HYPER_DIFF_DIR / "data" / "partnet" / "ins_seg_h5" / category / f"{split}-{index}-sem_seg_ids.json", 'w') as fout:
    json.dump(shape_segementation_id_mapping, fout)
