import torch
from torch.utils.data import Dataset

import os
import glob
import cv2

def img2patches(img, kernel, stride):
    """
    Divides an image into patches.

    Uses a sliding-window to extract patches from an image.

    Parameters
    ----------
    img : Tensor
        Input image
    kernel : int
        Size of the sliding-window.
    stride : int
        Stride of the sliding-window
    """
    patches = img.unfold(1, kernel, stride).unfold(2, kernel,stride)
    patches = patches.contiguous()

    return patches.view(patches.shape[0],-1,*patches.shape[3:]).permute(1,0,2,3)

def load_dataset(path, ext, patch_size, stride, transform=None):
    """
    Loads images from a directory.

    Loads the images and uses `img2patches' to extract patches.

    Parameters
    ----------
    path : string
        Path of the directory containing the images.
    ext : string
        Extension of the images.
    patch_size : int
        Size of the sliding-window for patch extraction.
    stride : int
        Stride of the sliding-window for patch extraction.
    transform : Transform
        Transform to apply to the whole image. Default None.

    Returns
    -------
    data_torch: Tensor
        Image patches.
    """
    imgs = glob.glob(os.path.join(path,"*.%s"%ext))
    data = []

    for path in imgs:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise OSError(f'Could not open image {path}')
        img = torch.tensor(img, dtype=torch.float).unsqueeze(0)/255
        if transform:
            img = transform(img)
        patches = img2patches(img, patch_size, stride)
        data.append(patches)
    data_torch = torch.Tensor(len(data), *data[0].shape)
    torch.cat(data, out=data_torch)

    return data_torch

class LazyDataset(Dataset):
    def __init__(self, root_dir, ext='png', transform=None):
        """
        Dataset class, loads images lazily.

        [TODO:description]

        Parameters
        ----------
        root_dir : string
            Path of the directory containing the images.
        ext : string
            File extension of the images.
        transform : Transform
            Transform to apply to images.
        """
        self.paths = glob.glob(os.path.join(root_dir,"*.%s"%ext))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise OSError(f'Could not open image {path}')
        img = torch.tensor(img, dtype=torch.float).unsqueeze(0)/255
        if self.transform:
            img = self.transform(img)
        return img

class ImgPatches(Dataset):
    def __init__(self, root_dir, patch_size, stride=1, transform=None,
            patch_transform=None):
        """
        Dataset class for image patches.

        [TODO:description]

        Parameters
        ----------
        root_dir : string
            Path of the directory containing the images.
        patch_size : int
            Size of the sliding-window for patch extraction.
        stride : int
            Stride of the sliding-window for patch extraction.
        transform : Transform
            Transform to apply to the whole image. Default None.
        patch_transform : Transform
            Transform to apply to image patches. Default None.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.patch_transform = patch_transform
        self.patch_size = patch_size
        self.stride = stride
        self.data = load_dataset(root_dir, "png", self.patch_size, self.stride,
                self.transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patch = self.data[idx]
        if self.patch_transform:
            patch = self.patch_transform(patch)
        return patch

class ConcatDatasets(Dataset):
    def __init__(self, *datasets):
        """
        Concatenates two datasets.

        [TODO:description]
        """
        self.datasets = datasets

    def __len__(self):
        return max(len(d) for d in self.datasets)

    def __getitem__(self, i):
        return tuple(d[i % len(d)] for d in self.datasets)

