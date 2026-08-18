import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ImageRestorationDataset(Dataset):

    def __init__(self, gt_dir, noisy_dir, filenames):
        self.gt_dir = gt_dir
        self.noisy_dir = noisy_dir
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]

        gt = np.load(
            os.path.join(self.gt_dir, name)
        ).astype(np.float32)

        noisy = np.load(
            os.path.join(self.noisy_dir, name)
        ).astype(np.float32)

        gt = torch.from_numpy(gt).unsqueeze(0)
        noisy = torch.from_numpy(noisy).unsqueeze(0)

        return noisy, gt