# import necessary modules
import numpy as np
import os
import pandas as pd
import random
import torch

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from .H5Dataset import H5Dataset


class AxialSliceDataset(Dataset):
    def __init__(self, config, mode="train", opt="opt"):
        self.dir_data = config["data_dir"]              # root directory for datasets
        self.fold = config["fold"]                      # 5-fold cross-validation index

        self.n_slices = config["n_slices"]              # number of axial slices to generate at once
        self.ps = config["ps"]                          # patch size
        self.random_shift = config["random_shift"]      # random offset in x- and y-direction from centerline

        self.bias = config["bias"]
        self.factor = config["factor"]
        self.normalize = config["normalize"]

        self.id_list = []
        self.label = {}

        self.sheet_optimal = {}                         # optimal contrast scans
        self.sheet_low = {}                             # below optimal contrast scans
        self.sheet_high = {}                            # above optimal contrast scans
        for dataset in config["datasets"]:
            sheet_ostia = pd.read_excel(os.path.join(self.dir_data, dataset, "ostia.xlsx"))

            # filter out unreliable stuff
            sheet_ostia = sheet_ostia.iloc[sheet_ostia.groupby("ID").apply(lambda x: x["std"].idxmin())]
            sheet_ostia = sheet_ostia.drop_duplicates(subset=["mu", "std"])
            sheet_ostia = sheet_ostia[sheet_ostia["std"] < 500]

            # separate into optimal and suboptimal attenuation
            sheet_optimal = sheet_ostia[sheet_ostia['mu'].between(300, 500)]
            sheet_low = sheet_ostia[sheet_ostia['mu'] <= 300]
            sheet_high = sheet_ostia[sheet_ostia['mu'] >= 500]

            self.sheet_optimal[dataset] = sheet_optimal
            self.sheet_low[dataset] = sheet_low
            self.sheet_high[dataset] = sheet_high
            for i, sheet in enumerate((sheet_low, sheet_optimal, sheet_high)):
                for id_ in list(sheet["ID"]):
                    image = id_
                    id_ = os.path.join(self.dir_data, dataset, "image", "image" + id_, image + ".h5")

                    self.id_list.append(id_)
                    self.label[id_] = int(i)            # 0 = below, 1 = optimal, 2 = above

        # cross-validation setup
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        x, y = np.array(self.id_list), np.array(list(self.label.values()))
        self.folds = []
        for train_ix, val_ix in kfold.split(x, y):
            self.folds.append([x[train_ix], x[val_ix]])

        # only select part of dataset that is either optimal or suboptimal
        if opt == "low":
            id_list = [k for k, v in self.label.items() if v == 0]
        elif opt == "opt":
            id_list = [k for k, v in self.label.items() if v == 1]
        elif opt == "high":
            id_list = [k for k, v in self.label.items() if v == 2]
        else:
            raise ValueError("opt must be one of ('low', 'opt', 'high')")

        # split to train/val
        if mode == "train":
            self.id_list = list(set(self.folds[self.fold][0]) & set(id_list))
        elif mode == "val":
            self.id_list = list(set(self.folds[self.fold][1]) & set(id_list))

    def norm(self, image):
        return (image + self.bias) / self.norm

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        # select and prepare
        image = H5Dataset(self.id_list[index], n_slices=self.n_slices)

        slice = image.get_axial_slice()
        slice = slice.transpose(2, 0, 1)

        if self.normalize:
            slice = self.normalize(slice)
        return torch.from_numpy(slice.copy()).float()


class AxialPatchDataset(AxialSliceDataset):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

    def __getitem__(self, index):
        # select and prepare
        image = H5Dataset(self.id_list[index], n_slices=self.n_slices)
        if random.random() < 0.5:
            patch, mask = image.get_axial_patch(self.ps)
        else:
            patch, mask = image.get_axial_centerline_patch(self.ps, self.random_shift)

        # augment
        if random.random() < 0.5:
            rot = random.randrange(1, 4)
            patch = np.rot90(patch, rot)
            mask = np.rot90(mask, rot)
        if random.random() < 0.5:
            flip = random.randrange(0, 2)
            patch = np.flip(patch, flip)
            mask = np.flip(mask, flip)

        patch = patch.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        if self.normalize:
            patch = self.normalize(patch)
        return torch.from_numpy(patch.copy()).float(), torch.from_numpy(mask.copy())
