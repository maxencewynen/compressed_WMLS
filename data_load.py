"""
adapted from https://github.com/Shifts-Project/shifts/tree/main/mswml
"""
import numpy as np
import os
from os.path import join as pjoin
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    AddChanneld, Compose, LoadImaged, RandCropByPosNegLabeld,
    ToTensord, NormalizeIntensityd, RandFlipd,
    RandRotate90d, RandShiftIntensityd, RandAffined, RandSpatialCropd,
    RandScaleIntensityd, Lambdad,)
import random
from sklearn.model_selection import train_test_split


def get_split(files, split, seed=1):
    assert split in ("train", "test", "val"), f"expected one of ('train', 'test', 'val') but got {split}"

    # Split the subjects into train, val, and test sets
    train_subjects, test_subjects = train_test_split(files, test_size=0.28, random_state=seed)
    train_subjects, val_subjects = train_test_split(train_subjects, test_size=0.25, random_state=seed)
    if split == "test": return test_subjects
    if split == "train": return train_subjects
    return val_subjects


def get_train_transforms():
    """ Get transforms for training on FLAIR images and ground truth:
    - Loads 3D images from Nifti file
    - Adds channel dimention
    - Normalises intensity
    - Applies augmentations
    - Crops out 32 patches of shape [96, 96, 96] that contain lesions
    - Converts to torch.Tensor()
    """

    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Lambdad(keys="label", func=lambda x: (x > 0). astype(np.uint8)),
            NormalizeIntensityd(keys=["image"], nonzero=True),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandCropByPosNegLabeld(keys=["image", "label"],
                                   label_key="label", image_key="image",
                                   spatial_size=(128, 128, 128), num_samples=32,
                                   pos=4, neg=1),
            RandSpatialCropd(keys=["image", "label"],
                             roi_size=(96, 96, 96),
                             random_center=True, random_size=False),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=(0, 1, 2)),
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1)),
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(1, 2)),
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 2)),
            RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'),
                        prob=1.0, spatial_size=(96, 96, 96),
                        rotate_range=(np.pi / 12, np.pi / 12, np.pi / 12),
                        scale_range=(0.1, 0.1, 0.1), padding_mode='border'),
            ToTensord(keys=["image", "label"]),
        ]
    )


def get_val_transforms():
    """ Get transforms for testing on FLAIR images and ground truth:
    - Loads 3D images and masks from Nifti file
    - Adds channel dimention
    - Applies intensity normalisation to scans
    - Converts to torch.Tensor()
    """
    return Compose(
        [
            LoadImaged(keys=("image", "label")),
            AddChanneld(keys=("image", "label")),
            Lambdad(keys="label", func=lambda x: (x > 0). astype(np.uint8)),
            NormalizeIntensityd(keys=("image",), nonzero=True),
            ToTensord(keys=("image", "label")),
        ]
    )


def get_train_dataloader(data_dir, num_workers, cache_rate=0.1):
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

    images = sorted([pjoin(data_dir, "images", i) for i in os.listdir(pjoin(data_dir, "images")) if i.endswith('FLAIR.nii.gz')])
    labels = sorted([pjoin(data_dir, "labels", i) for i in os.listdir(pjoin(data_dir, "labels")) if i.endswith('instances.nii.gz')])
    bms = sorted([pjoin(data_dir, "brainmasks", i) for i in os.listdir(pjoin(data_dir, "brainmasks")) if i.endswith('brainmask.nii.gz')])

    images = sorted(get_split(images, "train", seed=1))
    labels = sorted(get_split(labels, "train", seed=1))
    bms = sorted(get_split(bms, "train", seed=1))

    files = [
        {"image": image, "label": label, "brain_mask": bm}
        for image, label, bm in zip(images, labels, bms)
    ]
    assert len(images) == len(labels) == len(bms)

    print("Number of training files:", len(files))
    train_transforms = get_train_transforms()

    for f in files:
        f['subject'] = os.path.basename(f["label"])[:7]

    ds = CacheDataset(data=files, transform=train_transforms, cache_rate=cache_rate, num_workers=num_workers)
    return DataLoader(ds, batch_size=1, shuffle=True, num_workers=num_workers)


def get_val_dataloader(data_dir, num_workers, cache_rate=0.1, test=False):
    """
    Get dataloader for validation and testing. Either with or without brain masks.

    Args:
      data_dir: `str`, path to data directory (should contain img/ and labels/).
      num_workers:  `int`,  number of worker threads to use for parallel processing
                    of images
      cache_rate:  `float` in (0.0, 1.0], percentage of cached data in total.
      test: `bool`, whether to use the test split or the val split.
    Returns:
      monai.data.DataLoader() class object.
    """

    assert os.path.exists(data_dir), f"data_dir path does not exist ({data_dir})"

    images = sorted(
        [pjoin(data_dir, "images", i) for i in os.listdir(pjoin(data_dir, "images")) if i.endswith('FLAIR.nii.gz')])
    labels = sorted(
        [pjoin(data_dir, "labels", i) for i in os.listdir(pjoin(data_dir, "labels")) if i.endswith('instances.nii.gz')])
    bms = sorted([pjoin(data_dir, "brainmasks", i) for i in os.listdir(pjoin(data_dir, "brainmasks")) if
                  i.endswith('brainmask.nii.gz')])

    dirname = "test" if test else "val"
    images = sorted(get_split(images, dirname, seed=1))
    labels = sorted(get_split(labels, dirname, seed=1))
    bms = sorted(get_split(bms, dirname, seed=1))

    files = [
        {"image": image, "label": label, "brain_mask": bm}
        for image, label, bm in zip(images, labels, bms)
    ]
    assert len(images) == len(labels) == len(bms)

    print("Number of Validation files:", len(files))
    val_transforms = get_val_transforms()

    for f in files:
        f['subject'] = os.path.basename(f["label"])[:7]

    ds = CacheDataset(data=files, transform=val_transforms, cache_rate=cache_rate, num_workers=num_workers)
    return DataLoader(ds, batch_size=1, shuffle=not test, num_workers=num_workers)


def get_test_dataloader(data_dir, num_workers, cache_rate=0):
    """
    Get dataloader for validation and testing. Either with or without brain masks.

    Args:
      data_dir: `str`, path to data directory (should contain img/ and labels/).
      num_workers:  `int`,  number of worker threads to use for parallel processing
                    of images
      cache_rate:  `float` in (0.0, 1.0], percentage of cached data in total.
      test: `bool`, whether to use the test split or the val split.
    Returns:
      monai.data.DataLoader() class object.
    """
    return get_val_dataloader(data_dir, num_workers, cache_rate=cache_rate, test=True)


if __name__ == "__main__":
    dl = get_train_dataloader(data_dir="/home/mwynen/data/cusl_wml/all", num_workers=1)
    for x in dl:
        break
    breakpoint()
    # import nibabel as nib
    #
    # nib.save(nib.Nifti1Image(np.squeeze(x['label'][0].numpy()), np.eye(4)), 'label_deleteme.nii.gz')
    # nib.save(nib.Nifti1Image(np.squeeze(x['center_heatmap'][0].numpy()), np.eye(4)), 'heatmap_deleteme.nii.gz')
    # nib.save(nib.Nifti1Image(np.squeeze(x['offsets'][0].numpy()).transpose(1, 2, 3, 0), np.eye(4)),
    #          'com_reg_deleteme.nii.gz')
    # nib.save(nib.Nifti1Image(np.squeeze(x['image'][0].numpy()), np.eye(4)), 'image_deleteme.nii.gz')
