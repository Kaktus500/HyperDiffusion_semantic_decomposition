import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)  # TODO: Fix this for debug ...

import shutil

from google.cloud import storage

from helpers import HYPER_DIFF_DIR


def list_blobs(bucket_name: str):
    """Lists all the blobs in the bucket."""
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name)

    print("Blobs in bucket:")
    for blob in blobs:

        print(blob.name)

def download_blob(bucket_name: str, source_blob_name: str, destination_file_name: str):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    Path(destination_file_name).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(destination_file_name)

    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

def upload_blob(bucket_name: str, source_file_name: str, destination_blob_name: str):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(f"File {source_file_name} uploaded to {destination_blob_name}.")

def upload_folder(bucket_name: str, source_folder: str, destination_folder: str):
    """Uploads a folder to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    for file in Path(source_folder).rglob("*"):
        if file.is_file():
            blob = bucket.blob(destination_folder / file.relative_to(source_folder))
            blob.upload_from_filename(str(file))

def upload_folder_zipped(bucket_name: str, source_folder: str, destination_folder: str):
    """Uploads a folder to the bucket."""
    storage_client = storage.Client()
    if not Path(source_folder).exists():
        print(f"File {source_folder} does not exist")
        return
    if not Path(source_folder).with_suffix(".zip").exists():
        shutil.make_archive(source_folder, 'zip', source_folder)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(str(Path(destination_folder).with_suffix(".zip")))
    blob.upload_from_filename(str(Path(source_folder).with_suffix(".zip")))

if __name__ == "__main__":
    # add export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-file.json" to .bashrc for auth
    bucket_name = "adlcv-dataset-bucket"
    list_blobs(bucket_name)
    upload_blob(bucket_name, "/home/pauldelseith/HyperDiffusion_semantic_decomposition/siren/experiment_scripts/logs/chair_complete_baseline.zip", "chair_complete_baseline_l4_split_5_6.zip")

    # for i in range(9):
    #     local_data_path = HYPER_DIFF_DIR / "data" / "partnet" / "sem_seg_meshes" / f"Chair_100000_pc_occ_in_out_True_split_{i}.zip" 
    #     bucket_data_path = Path("partnet/Chair") / f"Chair_100000_pc_occ_in_out_True_split_{i}.zip"

    #     # upload_folder_zipped(bucket_name, str(local_data_path), str(bucket_data_path / "Chair_100000_pc_occ_in_out_True_split_8"))
    #     download_blob(bucket_name, str(bucket_data_path), str(local_data_path))
