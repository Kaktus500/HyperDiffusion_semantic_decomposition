from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent)) # TODO: Fix this for debug ...
from helpers import HYPER_DIFF_DIR
from huggingface_hub import snapshot_download
from datasets import get_dataset_split_names, load_dataset

if __name__ == "__main__":
    partnet_data_path = HYPER_DIFF_DIR.parent / "dataset_storage" / "partnet"
    if not partnet_data_path.exists():
        partnet_data_path.mkdir(parents=True)
    snapshot_download(repo_id="ShapeNet/PartNet-archive", repo_type="dataset", ignore_patterns=["*.md","*.gitattributes"], cache_dir=partnet_data_path)
    