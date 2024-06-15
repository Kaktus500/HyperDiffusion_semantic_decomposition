import csv
import glob
import math
import os

import igl
import matplotlib.colors as colors
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.ndimage
import scipy.special
import skimage
import skimage.filters
import skvideo.io
import torch
import trimesh
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


class PointCloud(Dataset):
    def __init__(self, path, file, classes, on_surface_points):
        super().__init__()
        print("Loading point cloud for ", path, file)

        self.labels = []
        self.coords = []
        self.occupancies = []

        for i, c in enumerate(classes):
            point_cloud = np.load(os.path.join(path, file + "_" + c + ".obj.npy"))
            self.coords.append(point_cloud[:, :3])
            self.occupancies.append(point_cloud[:, 3])
            self.labels.extend([i] * point_cloud.shape[0])

        self.coords = np.concatenate(self.coords, axis=0)
        self.occupancies = np.concatenate(self.occupancies, axis=0)
        self.labels = np.array(self.labels)
        self.on_surface_points = on_surface_points

    def __len__(self):
        return len(self.coords) // self.on_surface_points

    def __getitem__(self, idx):
        length = 200000
        idx_size = self.on_surface_points
        idx = np.random.randint(length, size=idx_size)

        if np.random.randint(2) == 0:
            idx += length

        coords = self.coords[idx]
        occs = self.occupancies[idx, None]
        labels = self.labels[idx]

        return {"coords": torch.from_numpy(coords).float()}, {
            "sdf": torch.from_numpy(occs)
        }, {"labels": torch.from_numpy(labels).long()}