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
    def __init__(
        self,
        path,
        file, # this is the base name of the file without the class name (e.g. "chair" instead of "chair_base.obj.npy")
        classes, # this is a list of class names (e.g. ["base", "back", "seat", "leg"])
        on_surface_points,
    ):
        super().__init__()
        print("Loading point cloud for ", path, file)


        ############################ This part is specific to our semantic decompositioning task ############################

        self.labels = [] # this will be the label

        point_clouds = [] # This contains a list of point clouds for each class. The list will be concatenated later.
        for i, c in enumerate(classes):
            point_cloud = np.load(
                os.path.join(path, file + "_" + c + ".obj.npy")
            ) # this has dimensionality (N, 4) where N is the number of points in the point cloud
            point_clouds.append(point_cloud)
            self.labels.extend([i] * point_cloud.shape[0]) # a class has e.g. 200000 points so we add 200000 labels for this class

        point_cloud = np.concatenate(point_clouds, axis=0)

        ################################################################################################################

        self.coords = point_cloud[:, :3]
        self.occupancies = point_cloud[:, 3]

        self.on_surface_points = on_surface_points

    def __len__(self):
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx): # this input idx doesn't do anything
        # length = self.coords.shape[0]
        # to only select points from a certain class we set this to the length of the class
        length = 200000
        idx_size = self.on_surface_points
        idx = np.random.randint(length, size=idx_size)

        if np.random.randint(2) == 0:
            idx += length

        coords = self.coords[idx]
        occs = self.occupancies[idx, None]
        labels = np.array(self.labels)[idx]

        return {"coords": torch.from_numpy(coords).float()}, {
            "sdf": torch.from_numpy(occs)
        }, {
            "labels": torch.from_numpy(labels).long()} # this is the label for the semantic decompositioning