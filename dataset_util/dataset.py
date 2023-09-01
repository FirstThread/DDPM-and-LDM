import torch
import json

import numpy as np
import matplotlib.pyplot as plt

from torchvision.datasets import VisionDataset
from torchvision.transforms import ToTensor
from PIL import Image
from pathlib import Path
from typing import Callable, Optional, Tuple

# Blueprint used: https://pytorch.org/vision/main/_modules/torchvision/datasets/cityscapes.html#Cityscapes
class BinsceneA(VisionDataset):
    """Dataset for auto-generated cluttered bin scenes.
    Args:
        root (string): Directory where the dataset is located.
        split (string, optional): The data split to use, ``full``, ``train`` or ``val`` 
        transform (callable, optional): A function/transform that takes the data
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the 
        target and transforms it.
        transforms (callable, optional): A function/transform that takes input data sample
            and its target as entry and returns a transformed version.
        premultiply_alpha: bool: Whether to remove background for rgb images.
    """
    def __init__(
        self,
        root: str,
        split: str = 'full',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        premultiply_alpha: bool = True,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.root = Path(root)
        self.split = split
        self.premultiply_alpha = premultiply_alpha
        self.to_tensor = ToTensor()
        
        info_dir = self.root / 'dataset_info'
        with open(info_dir / 'classes.json', "r") as fp:
            self.class_labels = json.load(fp)
        self.class_labels = np.array(self.class_labels, dtype=object)
        with open(info_dir / 'sample_ids.json', "r") as fp:
            self.filenames = json.load(fp)
        self.filenames = np.array(self.filenames, dtype=object)
        if self.split != 'full':
            n_samples = len(self.filenames)
            indices = np.arange(n_samples)
            split_frac = int(np.ceil(0.8 * n_samples))
            train_idx, valid_idx = indices[:split_frac], indices[split_frac:]
            if self.split == 'train' or self.split == 'training':
                self.filenames = self.filenames[train_idx]
            elif self.split == 'val' or self.split == 'valid' or self.split == 'validation':
                self.filenames = self.filenames[valid_idx]
            else:
                raise NotImplementedError
                             
    def __len__(self) -> int:
        return len(self.filenames)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # get image file path and load image
        data_filepath = self.root / 'rgb' / (self.filenames[idx] + '.png')
        if self.premultiply_alpha:
            img = self.to_tensor(Image.open(data_filepath).convert('RGBa').convert('RGBA').convert('RGB'))
        else:
            img = self.to_tensor(Image.open(data_filepath).convert('RGB'))
        # get visibility vector filepath and load
        target_filepath = self.root / 'visib' / (self.filenames[idx] +'.pt')
        target = torch.load(target_filepath)
        # transform both as needed
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target
    
    def show(self, img=None, target=None, ax=None, param_dict={}):
        out_ax, out_desc = None, None
        if img is not None:
            if ax is None:
                f, out_ax = plt.subplots()
                out_ax.imshow(img.permute(1,2,0), **param_dict)
            else:
                out_ax = ax.imshow(img.permute(1,2,0), **param_dict)
        if target is not None:
            out_desc = 'Visible objects are'
            target_labels = self.class_labels[torch.nonzero(target).squeeze()]
            target_labels = target_labels[np.argsort(target_labels)]
            for label in target_labels[:-2]:
                out_desc += ' ' + label + ','
            out_desc += ' ' + target_labels[-2] + ' and ' + target_labels[-1] + '.'
        return out_ax, out_desc
