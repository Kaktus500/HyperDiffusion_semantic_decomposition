from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent)) # TODO: Fix this for debug ...
from helpers import HYPER_DIFF_DIR
from huggingface_hub import hf_hub_download
from datasets import get_dataset_split_names, load_dataset

if __name__ == "__main__":
    hf_hub_download(repo_id="ShapeNet/PartNet-archive", filename="data_v0_chunk.zip", local_dir=HYPER_DIFF_DIR / "data" / "partnet", repo_type="dataset")