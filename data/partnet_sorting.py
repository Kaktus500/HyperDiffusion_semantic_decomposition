import json
from pathlib import Path
import sys
import shutil
from typing import Dict, List
sys.path.append(str(Path(__file__).resolve().parent.parent)) # TODO: Fix this for debug ...
from helpers import HYPER_DIFF_DIR

if __name__ == "__main__":
    with open(HYPER_DIFF_DIR / "data" / "partnet_class_mapping.json", "r") as f:
        partnet_id_mapping: Dict[str, List[int]] = json.load(f)
    for name, ids in partnet_id_mapping.items():
        if name.lower() == "chair":
            for instance in ids:
                file_path = Path.home() / "dataset_storage" / "partnet" / "data_v0" / f"{instance}"
                copy_file_path = HYPER_DIFF_DIR / "data" / name.lower() / f"{instance}"
                if copy_file_path.exists():
                    continue
                shutil.copytree(file_path, copy_file_path)
