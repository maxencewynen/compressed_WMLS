import argparse
import os
import nibabel as nib
import pandas as pd
import numpy as np
import nibabel as nib
from postprocess import *


def dice_metric(ground_truth, predictions):
    """
    Compute Dice coefficient for a single example.
    Args:
      ground_truth: `numpy.ndarray`, binary ground truth segmentation target,
                     with shape [W, H, D].
      predictions:  `numpy.ndarray`, binary segmentation predictions,
                     with shape [W, H, D].
    Returns:
      Dice coefficient overlap (`float` in [0.0, 1.0])
      between `ground_truth` and `predictions`.
    """
    # Calculate intersection and union of y_true and y_predict
    intersection = np.sum(predictions * ground_truth)
    union = np.sum(predictions) + np.sum(ground_truth)

    # Calculate dice metric
    if intersection == 0.0 and union == 0.0:
        dice = 1.0
    else:
        dice = (2. * intersection) / union

    return dice


def compute_dices_from_paths(gt_paths, pred_paths):
    import nibabel as nib
    assert len(gt_paths) == len(pred_paths), f"{len(gt_paths)}, {len(pred_paths)}"
    avg_dsc = 0
    print("Computing dice on the dataset...")
    for gt_path, pred_path in zip(gt_paths, pred_paths):
        # Load nifti files
        gt_img = nib.load(gt_path)
        pred_img = nib.load(pred_path)

        # Get data from nifti file
        gt = gt_img.get_fdata()
        pred = pred_img.get_fdata()

        avg_dsc += dice_metric(gt, pred)

    avg_dsc /= len(gt_paths)
    print(f"The dice score of the dataset averaged over all the subjects is {avg_dsc}")


def dice_norm_metric(ground_truth, predictions):
    """
    Compute Normalised Dice Coefficient (nDSC),
    False positive rate (FPR),
    False negative rate (FNR) for a single example.

    Args:
      ground_truth: `numpy.ndarray`, binary ground truth segmentation target,
                     with shape [H, W, D].
      predictions:  `numpy.ndarray`, binary segmentation predictions,
                     with shape [H, W, D].
    Returns:
      Normalised dice coefficient (`float` in [0.0, 1.0]),
      False positive rate (`float` in [0.0, 1.0]),
      False negative rate (`float` in [0.0, 1.0]),
      between `ground_truth` and `predictions`.
    """

    # Reference for normalized DSC
    r = 0.001
    # Cast to float32 type
    gt = ground_truth.astype("float32")
    seg = predictions.astype("float32")
    im_sum = np.sum(seg) + np.sum(gt)
    if im_sum == 0:
        return 1.0
    else:
        if np.sum(gt) == 0:
            k = 1.0
        else:
            k = (1 - r) * np.sum(gt) / (r * (len(gt.flatten()) - np.sum(gt)))
        tp = np.sum(seg[gt == 1])
        fp = np.sum(seg[gt == 0])
        fn = np.sum(gt[seg == 0])
        fp_scaled = k * fp
        dsc_norm = 2. * tp / (fp_scaled + 2. * tp + fn)
        return dsc_norm


def compute_metrics(args):
    # Check if prediction and reference folders exist
    if not os.path.exists(args.pred_path) or not os.path.exists(args.ref_path):
        print("Either prediction or reference path doesn't exist!")
        return

    metrics_dict = {"Subject_ID": [], "File": [], "DSC": [], "nDSC": []}

    ref_dir = args.ref_path
    
    for pred_file in sorted(os.listdir(args.pred_path)):
        if pred_file.endswith("binary.nii.gz"):
            print(pred_file)
            subj_id = pred_file.split("_ses")[0].split("sub-")[-1]  # Extracting subject ID
            ref_file = "sub-" + subj_id + "_ses-01_mask-instances.nii.gz"
            ref_file_path = os.path.join(ref_dir, ref_file)

            pred_img = nib.load(os.path.join(args.pred_path, pred_file)).get_fdata()
            ref_img = nib.load(ref_file_path)
            voxel_size = ref_img.header.get_zooms()
            ref_img = remove_small_lesions_from_binary_segmentation((ref_img.get_fdata()>0).astype(np.uint8), voxel_size)

            if not set(np.unique(pred_img)) == {0, 1}:
                print("[WARNING] Prediction image should be binary. Skipping this subject...")
                continue

            dsc = dice_metric(ref_img, (pred_img > 0).astype(np.uint8))
            ndsc = dice_norm_metric(ref_img, (pred_img > 0).astype(np.uint8))
            metrics_dict["DSC"].append(dsc)
            metrics_dict["nDSC"].append(ndsc)
            metrics_dict["Subject_ID"].append(subj_id)
            metrics_dict["File"].append(pred_file)

    model_name = os.path.basename(os.path.dirname(args.pred_path))
    # Convert dictionary to dataframe and save as CSV
    df = pd.DataFrame(metrics_dict)
    dd = "test" if args.test else "val"
    df.to_csv(os.path.join(args.pred_path, f"metrics_{model_name}_{dd}.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute segmentation metrics.")
    parser.add_argument("--pred_path", required=True, help="Path to the directory with prediction files.")
    parser.add_argument("--ref_path", required=True,
                        help="Path to the directory with reference files (containing val/ and test/).")
    parser.add_argument("--test", action="store_true", help="Wether to use the test data or not. Default is val data.")

    args = parser.parse_args()
    compute_metrics(args)
