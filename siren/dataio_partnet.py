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
    def __init__(self, path, file, classes: list, on_surface_points, *, split_shapes: bool = True):
        super().__init__()
        print("Loading point cloud for ", path, file)

        self.labels = []
        self.coords = []
        self.occupancies = []

        classes = sorted(classes)
        self.nr_classes = len(classes)

        # assign labels to classes
        def _get_label(c):
            if c == "arm":
                return 0
            elif c == "back":
                return 1
            elif c == "base":
                return 2
            elif c == "seat":
                return 3

        for c in classes:
            point_cloud = np.load(os.path.join(path, file + "_" + c + ".obj.npy"))
            self.coords.append(point_cloud[:, :3])
            self.occupancies.append(point_cloud[:, 3])
            if split_shapes:
                self.labels.extend([_get_label(c)] * point_cloud.shape[0])
            else:
                # always assign same label
                self.labels.extend([4] * point_cloud.shape[0]) # TODO: Seems that 4 might be wrong here?

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

        # each class has 200k points
        # we sample from one random class for each batch
        i = np.random.randint(self.nr_classes)
        idx += length * i

        coords = self.coords[idx]
        occs = self.occupancies[idx, None]
        labels = self.labels[idx]

        return {"coords": torch.from_numpy(coords).float()}, {
            "sdf": torch.from_numpy(occs)
        }, {"labels": torch.from_numpy(labels).int()}