# import necessary modules
import h5py
import numpy as np
import random


class H5Dataset:
    def __init__(self, h5_path, n_slices):
        self.pad_z = (n_slices - 1) // 2
        self.path = h5_path

        with h5py.File(self.path, 'r') as file:
            self.image = file['ccta']['ccta']
            self.shape = self.image.shape

            # attributes
            self.spacing = self.image.attrs['spacing']
            self.offset = self.image.attrs['offset']
            self.ctl_points = self.image.attrs['ctl_points']

    def get_axial_slice(self, z=None):
        if not z:
            z = random.randrange(self.pad_z, self.shape[2] - self.pad_z)
        return self.image[:, :, z - self.pad_z: z + 1 + self.pad_z]

    def get_axial_patch(self, ps=64):
        pad = ps // 2

        # extract patch
        x, y, z = random.randrange(0, self.shape[0] - pad*2),\
                  random.randrange(0, self.shape[1] - pad*2),\
                  random.randrange(0, self.shape[2] - self.pad_z*2)
        bbox = [[x, x+pad*2], [y, y+pad*2], [z, z+1+self.pad_z*2]]

        slice_ = self.image[bbox[0][0]:bbox[0][1],
                            bbox[1][0]:bbox[1][1],
                            bbox[2][0]:bbox[2][1]]
        mask_ = np.full(slice_.shape, False)

        return slice_, mask_

    def get_axial_centerline_patch(self, ps=64, rs=0):
        pad = ps // 2
        coords = [pad - 1, pad - 1, self.pad_z]

        # extract patch
        index = random.randrange(0, self.ctl_points.shape[0])
        x, y, z = np.round((self.ctl_points[index, :3] - self.offset) / self.spacing).astype(int)

        # random shift x and y
        bbox = [[x-pad, x+pad], [y-pad, y+pad], [z-self.pad_z, z+1+self.pad_z]]
        if rs:
            shift_x = random.randrange(-rs, rs)
            shift_y = random.randrange(-rs, rs)
            for i, shift in enumerate([shift_x, shift_y]):
                bbox[i][0] += shift
                bbox[i][1] += shift
                coords[i] -= shift

        # make sure to operate within bounds
        for i, bounds in enumerate(bbox):
            if bounds[0] < 0:
                bounds[1] -= bounds[0]
                bounds[0] -= bounds[0]
                coords[i] += bounds[0]
            if bounds[1] > self.shape[i]:
                bounds[0] -= bounds[1] - self.shape[i]
                bounds[1] -= bounds[1] - self.shape[i]
                coords[i] += bounds[1] - self.shape[i]

        slice_ = self.image[bbox[0][0]:bbox[0][1],
                            bbox[1][0]:bbox[1][1],
                            bbox[2][0]:bbox[2][1]]
        mask_ = np.full(slice_.shape, False)
        mask_[coords[0], coords[1], coords[2]] = True

        return slice_, mask_
