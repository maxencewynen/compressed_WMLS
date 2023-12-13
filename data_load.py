"""
adapted from https://github.com/Shifts-Project/shifts/tree/main/mswml
"""
import numpy as np
import os
from os.path import join as pjoin
from glob import glob
import re
from monai.data import CacheDataset, DataLoader
from monai.data.meta_tensor import MetaTensor
from monai.transforms import (
    AddChanneld, Compose, LoadImaged, RandCropByPosNegLabeld,
    Spacingd, ToTensord, NormalizeIntensityd, RandFlipd,
    RandRotate90d, RandShiftIntensityd, RandAffined, RandSpatialCropd,
    RandScaleIntensityd, RandSpatialCropSamplesd, ConcatItemsd, Lambdad,
    MaskIntensityd)
from scipy import ndimage
import torch
from typing import Callable
from monai.transforms import MapTransform
from scipy.ndimage import center_of_mass
from monai.config import KeysCollection
from copy import deepcopy
import nibabel as nib
import torch.nn.functional as F
from postprocess import remove_small_lesions_from_instance_segmentation

DISCARDED_SUBJECTS = []
QUANTITATIVE_SEQUENCES = ["T1map"]


class Printer(Callable):
    def __init__(self, string):
        self.string = string

    def __call__(self, arg):
        if type(arg) == str:
            print(self.string, arg)
        elif type(arg) == torch.Tensor or type(arg) == torch.FloatTensor or type(arg) == np.ndarray:
            print(self.string, "(Shape =", arg.shape, ")")
        else:
            print(self.string, f"({type(arg)})")
        return arg



class SaveImageKeysd:
    def __init__(self, keys, output_dir, suffix=""):
        self.keys = keys
        self.output_dir = output_dir
        self.suffix = suffix

    def __call__(self, data):
        for key in self.keys:
            image = deepcopy(data[key])

            if key == "center_heatmap":
                image = torch.from_numpy(image) if type(image) == np.ndarray else image
                nms_padding = (3 - 1) // 2
                ctr_hmp = F.max_pool3d(image, kernel_size=3, stride=1, padding=nms_padding)
                ctr_hmp[ctr_hmp != image] = 0
                ctr_hmp[ctr_hmp > 0] = 1
                image = ctr_hmp
            if type(image) == torch.Tensor or type(image) == MetaTensor:
                image = image.cpu().numpy()
            image = np.squeeze(image)
            squeeze_dim = 4 if key == "offsets" else 3
            while len(image.shape) > squeeze_dim:
                image = image[0,:]

            if key == "offsets":
                image = image.transpose(1,2,3,0) #itksnap readable

            if self.suffix != "":
                filename = key + "_" + self.suffix + ".nii.gz"
            else:
                filename = key + ".nii.gz"
            filename = os.path.join(self.output_dir, filename)
            nib.save(nib.Nifti1Image(image, np.eye(4)), filename)
        return data


class Printerd:
    def __init__(self, keys, message=""):
        self.keys = keys
        self.message = message

    def __call__(self, data):
        for key in self.keys:
            image = data[key]
            # print(self.message, np.unique(image))
            print(self.message, key, image.dtype)
        return data


class BinarizeInstancesd(MapTransform):
    def __init__(self, keys, out_key="label"):
        self.keys = keys
        self.out_key = out_key

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            out_key = self.out_key + "_" + key if len(self.keys) > 1 else self.out_key
            image = deepcopy(data[key])
            image[image > 0] = 1
            d[out_key] = image.astype(np.uint8)
        return d


class LesionOffsetTransformd(MapTransform):
    """
    A MONAI transform to compute the offsets for each voxel from the center of mass of its lesion.
    """

    def __init__(self, keys: KeysCollection, allow_missing_keys=False, remove_small_lesions=False, l_min=14):
        """
        Args:
            key (str): the key corresponding to the desired data in the dictionary to apply the transformation.
        """
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        if type(keys) == list and len(keys) > 1:
            raise Exception("This transform should only be used with 1 key.")
        self.remove_small_lesions = remove_small_lesions
        self.l_min = l_min

    def __call__(self, data):
        d = dict(data)
        voxel_size = tuple(data[[k for k in list(data.keys()) if "_meta_dict" in k][0]]['pixdim'][1:4])
        for key in self.key_iterator(d):
            com_gt, com_reg = self.make_offset_matrices(d[key], voxel_size=voxel_size)
            d["center_heatmap"] = com_gt
            d["offsets"] = com_reg
            d["label"] = (d[key] > 0).astype(np.uint8)
        return d

    def make_offset_matrices(self, data, sigma=2, voxel_size=(1, 1, 1)):
        # Define 3D Gaussian function
        def gaussian_3d(x, y, z, cx, cy, cz, sigma):
            return np.exp(-((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (2 * sigma ** 2))

        data = np.squeeze(data)
        if self.remove_small_lesions:
            data = remove_small_lesions_from_instance_segmentation(data, voxel_size=voxel_size, l_min=self.l_min)

        heatmap = np.zeros_like(data, dtype=np.float32)
        offset_x = np.zeros_like(data, dtype=np.float32)
        offset_y = np.zeros_like(data, dtype=np.float32)
        offset_z = np.zeros_like(data, dtype=np.float32)

        # Create coordinate grids
        x_grid, y_grid, z_grid = np.meshgrid(np.arange(data.shape[0]),
                                             np.arange(data.shape[1]),
                                             np.arange(data.shape[2]),
                                             indexing='ij')

        # Get all unique lesion IDs (excluding zero which is typically background)
        lesion_ids = np.unique(data)[1:]

        # For each lesion id
        for lesion_id in lesion_ids:
            # Get binary mask for the current lesion
            mask = (data == lesion_id)

            # Compute the center of mass of the lesion
            cx, cy, cz = center_of_mass(mask)

            # Compute heatmap values using broadcasting
            current_gaussian = gaussian_3d(x_grid, y_grid, z_grid, cx, cy, cz, sigma)

            # Update heatmap with the maximum value encountered so far at each voxel
            heatmap = np.maximum(heatmap, current_gaussian)

            # Update offset matrices
            offset_x[mask] = cx - x_grid[mask]
            offset_y[mask] = cy - y_grid[mask]
            offset_z[mask] = cz - z_grid[mask]

        return np.expand_dims(heatmap, axis=0).astype(np.float32), \
            np.stack([offset_x, offset_y, offset_z], axis=0).astype(np.float32)


def get_train_transforms(I=['FLAIR'], apply_mask=None):
    """ Get transforms for training on FLAIR images and ground truth:
    - Loads 3D images from Nifti file
    - Adds channel dimention
    - Normalises intensity
    - Applies augmentations
    - Crops out 32 patches of shape [96, 96, 96] that contain lesions
    - Converts to torch.Tensor()
    """

    masks = ["instance_mask"]  # , "brain_mask"]
    non_label_masks = []
    if apply_mask:
        masks += [apply_mask]
        non_label_masks += [apply_mask]

    other_keys = ["label", "center_heatmap", "offsets"]

    non_quantitative_images = [i for i in I if i not in QUANTITATIVE_SEQUENCES]

    transform_list = [
        LoadImaged(keys=I + masks),
        AddChanneld(keys=I + masks),
        # Lambdad(keys=non_label_masks, func=lambda x: x.astype(np.uint), allow_missing_keys=True),
        NormalizeIntensityd(keys=non_quantitative_images, nonzero=True),
        RandShiftIntensityd(keys=non_quantitative_images, offsets=0.1, prob=1.0),
        RandScaleIntensityd(keys=non_quantitative_images, factors=0.1, prob=1.0),
        BinarizeInstancesd(keys=masks),
    ]
    if apply_mask:
        transform_list += [MaskIntensityd(keys=I + masks, mask_key=apply_mask)]

    transform_list += [
        RandCropByPosNegLabeld(keys=I + ["instance_mask", "label"],
                               label_key="label", image_key=I[0],
                               spatial_size=(128, 128, 128), num_samples=32,
                               pos=4, neg=1),
        RandSpatialCropd(keys=I + ["instance_mask", "label"],
                         roi_size=(96, 96, 96),
                         random_center=True, random_size=False),
        RandFlipd(keys=I + ["instance_mask", "label"], prob=0.5, spatial_axis=(0, 1, 2)),
        RandRotate90d(keys=I + ["instance_mask", "label"], prob=0.5, spatial_axes=(0, 1)),
        RandRotate90d(keys=I + ["instance_mask", "label"], prob=0.5, spatial_axes=(1, 2)),
        RandRotate90d(keys=I + ["instance_mask", "label"], prob=0.5, spatial_axes=(0, 2)),
        RandAffined(keys=I + ["instance_mask", "label"],
                    mode=tuple(['bilinear'] * len(I)) + ('nearest', 'nearest'),
                    prob=1.0, spatial_size=(96, 96, 96),
                    rotate_range=(np.pi / 12, np.pi / 12, np.pi / 12),
                    scale_range=(0.1, 0.1, 0.1), padding_mode='border'),
        LesionOffsetTransformd(keys="instance_mask"),
        ToTensord(keys=I + other_keys),
        ConcatItemsd(keys=I, name="image", dim=0)
    ]
    # transform.set_random_state(seed=seed)

    return Compose(transform_list)


def get_val_transforms(I=['FLAIR'], bm=False, apply_mask=None):
    """ Get transforms for testing on FLAIR images and ground truth:
    - Loads 3D images and masks from Nifti file
    - Adds channel dimention
    - Applies intensity normalisation to scans
    - Converts to torch.Tensor()
    """
    other_keys = ["instance_mask", "brain_mask"] if bm else ["instance_mask"]
    other_keys = other_keys + [apply_mask] if apply_mask else other_keys

    non_quantitative_images = [i for i in I if i not in QUANTITATIVE_SEQUENCES]

    transforms = [
        LoadImaged(keys=I + other_keys),
        AddChanneld(keys=I + other_keys),
        # Lambdad(keys=["label"], func=lambda x: (x>0).astype(int) ),
    ]
    transforms = transforms + [Lambdad(keys=["brain_mask"], func=lambda x: x.astype(np.uint8))] if bm else transforms
    transforms = transforms + [MaskIntensityd(keys=I, mask_key=apply_mask)] if apply_mask else transforms

    transforms += [
        Lambdad(keys=["brain_mask"], func=lambda x: x.astype(np.uint8)),
        NormalizeIntensityd(keys=non_quantitative_images, nonzero=True),
        LesionOffsetTransformd(keys="instance_mask", remove_small_lesions=True),
        ToTensord(keys=I + other_keys + ["label", "center_heatmap", "offsets"]),
        ConcatItemsd(keys=I, name="image", dim=0)
    ]
    return Compose(transforms)


def get_train_dataloader(data_dir, num_workers, cache_rate=0.1, seed=1, I=['FLAIR'], apply_mask=None, cp_factor=0):
    """
    Get dataloader for training
    Args:
      data_dir: `str`, path to data directory (should contain img/ and labels/).
      num_workers:  `int`,  number of worker threads to use for parallel processing
                    of images
      cache_rate:  `float` in (0.0, 1.0], percentage of cached data in total.
      I: `list`, list of modalities to include in the data loader.
    Returns:
      monai.data.DataLoader() class object.
    """
    assert os.path.exists(data_dir), f"data_dir path does not exist ({data_dir})"
    assert apply_mask is None or type(apply_mask) == str
    traindir = "train" if cp_factor == 0 else f"train_scp-f{cp_factor}_cl5"
    img_dir = pjoin(data_dir, traindir, "images")
    lbl_dir = pjoin(data_dir, traindir, "labels")
    bm_path = pjoin(data_dir, "train", "brainmasks")
    mask_path = None if not apply_mask else pjoin(data_dir, "train", apply_mask)

    # Collect all modality images sorted
    all_modality_images = {}
    all_modality_images = {
        i: [
            pjoin(img_dir, s)
            for s in sorted(list(os.listdir(img_dir)))
            if s.endswith(i + ".nii.gz") and not any(subj in s for subj in DISCARDED_SUBJECTS)
        ]
        for i in I
    }
    for modality in I:
        for j in range(len([j for j in I if j == modality])):
            if j == 0: continue
            all_modality_images[modality + str(j)] = all_modality_images[modality]

    # Check all modalities have same length
    assert all(len(x) == len(all_modality_images[I[0]]) for x in
               all_modality_images.values()), "All modalities must have the same number of images"

    # Collect all corresponding ground truths
    maskname = "mask-instances"
    segs = [pjoin(lbl_dir, f) for f in sorted(list(os.listdir(lbl_dir))) if f.endswith(maskname + ".nii.gz")]

    assert len(all_modality_images[I[0]]) == len(
        segs), "Number of multi-modal images and ground truths must be the same"

    files = []

    bms = [pjoin(bm_path, f) for f in sorted(list(os.listdir(bm_path))) if f.endswith("brainmask.nii.gz")]
    if not apply_mask:
        assert len(all_modality_images[I[0]]) == len(segs) == len(
            bms), f"Some files must be missing: {[len(all_modality_images[I[0]]), len(segs), len(bms)]}"

        for i in range(len(segs)):
            file_dict = {"instance_mask": segs[i], "brain_mask": bms[i]}
            for modality in all_modality_images.keys():  # in I:
                file_dict[modality] = all_modality_images[modality][i]
            files.append(file_dict)

    else:
        masks = [pjoin(mask_path, f) for f in sorted(list(os.listdir(mask_path))) if f.endswith(".nii.gz")]
        assert len(all_modality_images[I[0]]) == len(segs) == len(bms) == len(masks), \
            f"Some files must be missing: {[len(all_modality_images[I[0]]), len(segs), len(bms), len(masks)]}"

        for i in range(len(segs)):
            file_dict = {"instance_mask": segs[i], "brain_mask": bms[i], apply_mask: masks[i]}
            for modality in all_modality_images.keys():  # in I:
                file_dict[modality] = all_modality_images[modality][i]
            files.append(file_dict)

    print("Number of training files:", len(files))
    train_transforms = get_train_transforms(list(all_modality_images.keys()), apply_mask=apply_mask)  # I

    for f in files:
        f['subject'] = os.path.basename(f["instance_mask"])[:7]

    ds = CacheDataset(data=files, transform=train_transforms, cache_rate=cache_rate, num_workers=num_workers)
    return DataLoader(ds, batch_size=1, shuffle=True, num_workers=num_workers)


def get_val_dataloader(data_dir, num_workers, cache_rate=0.1, I=['FLAIR'], test=False, apply_mask=None):
    """
    Get dataloader for validation and testing. Either with or without brain masks.

    Args:
      data_dir: `str`, path to data directory (should contain img/ and labels/).
      num_workers:  `int`,  number of worker threads to use for parallel processing
                    of images
      cache_rate:  `float` in (0.0, 1.0], percentage of cached data in total.
      bm_path:   `None|str`. If `str`, then defines path to directory with
                 brain masks. If `None`, dataloader does not return brain masks.
      I: `list`, list of I to include in the data loader.
    Returns:
      monai.data.DataLoader() class object.
    """

    assert os.path.exists(data_dir), f"data_dir path does not exist ({data_dir})"
    assert apply_mask is None or type(apply_mask) == str
    img_dir = pjoin(data_dir, "val", "images") if not test else pjoin(data_dir, "test", "images")
    lbl_dir = pjoin(data_dir, "val", "labels") if not test else pjoin(data_dir, "test", "labels")
    bm_path = pjoin(data_dir, "val", "brainmasks") if not test else pjoin(data_dir, "test", "brainmasks")
    if not apply_mask:
        mask_path = None
    else:
        mask_path = pjoin(data_dir, "val", apply_mask) if not test else pjoin(data_dir, "test", apply_mask)

    # Collect all modality images sorted
    all_modality_images = {}
    all_modality_images = {
        i: [
            pjoin(img_dir, s)
            for s in sorted(list(os.listdir(img_dir)))
            if s.endswith(i + ".nii.gz") and not any(subj in s for subj in DISCARDED_SUBJECTS)
        ]
        for i in I
    }
    for modality in I:
        for j in range(len([j for j in I if j == modality])):
            if j == 0: continue
            all_modality_images[modality + str(j)] = all_modality_images[modality]

    # Check all modalities have same length
    assert all(len(x) == len(all_modality_images[I[0]]) for x in
               all_modality_images.values()), "All modalities must have the same number of images"

    # Collect all corresponding ground truths
    maskname = "mask-instances"
    segs = [pjoin(lbl_dir, f) for f in sorted(list(os.listdir(lbl_dir))) if f.endswith(maskname + ".nii.gz")]

    assert len(all_modality_images[I[0]]) == len(
        segs), "Number of multi-modal images and ground truths must be the same"

    files = []
    if bm_path is not None:
        bms = [pjoin(bm_path, f) for f in sorted(list(os.listdir(bm_path))) if f.endswith("brainmask.nii.gz")]
        if not apply_mask:
            assert len(all_modality_images[I[0]]) == len(segs) == len(
                bms), f"Some files must be missing: {[len(all_modality_images[I[0]]), len(segs), len(bms)]}"

            for i in range(len(segs)):
                file_dict = {"instance_mask": segs[i], "brain_mask": bms[i]}
                for modality in all_modality_images.keys():  # in I:
                    file_dict[modality] = all_modality_images[modality][i]
                files.append(file_dict)

        else:
            masks = [pjoin(mask_path, f) for f in sorted(list(os.listdir(mask_path))) if f.endswith(".nii.gz")]
            assert len(all_modality_images[I[0]]) == len(segs) == len(bms) == len(masks), \
                f"Some files must be missing: {[len(all_modality_images[I[0]]), len(segs), len(bms), len(masks)]}"

            for i in range(len(segs)):
                file_dict = {"instance_mask": segs[i], "brain_mask": bms[i], apply_mask: masks[i]}
                for modality in all_modality_images.keys():  # in I:
                    file_dict[modality] = all_modality_images[modality][i]
                files.append(file_dict)

        val_transforms = get_val_transforms(list(all_modality_images.keys()), bm=True, apply_mask=apply_mask)

    else:
        if not apply_mask:
            assert len(all_modality_images[I[0]]) == len(
                segs), f"Some files must be missing: {[len(all_modality_images[I[0]]), len(segs)]}"
            for i in range(len(segs)):
                file_dict = {"instance_mask": segs[i]}
                for modality in all_modality_images.keys():  # in I:
                    file_dict[modality] = all_modality_images[modality][i]
                files.append(file_dict)
        else:
            bms = [pjoin(bm_path, f) for f in sorted(list(os.listdir(bm_path))) if f.endswith("brainmask.nii.gz")]
            masks = [pjoin(mask_path, f) for f in sorted(list(os.listdir(mask_path))) if f.endswith(".nii.gz")]
            assert len(all_modality_images[I[0]]) == len(segs) == len(bms) == len(masks), \
                f"Some files must be missing: {[len(all_modality_images[I[0]]), len(segs), len(bms), len(masks)]}"

            for i in range(len(segs)):
                file_dict = {"instance_mask": segs[i], "brain_mask": bms[i], apply_mask: masks[i]}
                for modality in all_modality_images.keys():  # in I:
                    file_dict[modality] = all_modality_images[modality][i]
                files.append(file_dict)
        val_transforms = get_val_transforms(list(all_modality_images.keys()), apply_mask=apply_mask)

    if test:
        print("Number of test files:", len(files))
    else:
        print("Number of validation files:", len(files))
    for f in files:
        f['subject'] = os.path.basename(f["instance_mask"])[:7]

    ds = CacheDataset(data=files, transform=val_transforms, cache_rate=cache_rate, num_workers=num_workers)
    return DataLoader(ds, batch_size=1, shuffle=True, num_workers=num_workers)


if __name__ == "__main__":
    dl = get_train_dataloader(data_dir="/home/mwynen/data/bxl", num_workers=1, I=["FLAIR"])
    for x in dl:
        break
    breakpoint()
    import nibabel as nib

    nib.save(nib.Nifti1Image(np.squeeze(x['label'][0].numpy()), np.eye(4)), 'label_deleteme.nii.gz')
    nib.save(nib.Nifti1Image(np.squeeze(x['center_heatmap'][0].numpy()), np.eye(4)), 'heatmap_deleteme.nii.gz')
    nib.save(nib.Nifti1Image(np.squeeze(x['offsets'][0].numpy()).transpose(1, 2, 3, 0), np.eye(4)),
             'com_reg_deleteme.nii.gz')
    nib.save(nib.Nifti1Image(np.squeeze(x['image'][0].numpy()), np.eye(4)), 'image_deleteme.nii.gz')
