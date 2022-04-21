import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode

import os
import glob

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

    for i, img_path in enumerate(imgs):
        img = read_image(img_path, ImageReadMode.GRAY).float()/255
        
        if transform:
            img = transform(img)

        patches = img2patches(img, patch_size, stride)
        data.append(patches)
    data_torch = torch.Tensor(len(data), *data[0].shape)
    torch.cat(data, out=data_torch)

    return data_torch
        
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

