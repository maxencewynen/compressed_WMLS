import argparse
import os
from monai.inferers import sliding_window_inference
from model import *
from monai.data import write_nifti
import numpy as np
from data_load import get_val_dataloader
from postprocess import remove_small_lesions_from_binary_segmentation
from metrics import dice_metric, dice_norm_metric
import nibabel as nib
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Get all command line arguments.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# save options
parser.add_argument('--path_pred', type=str, required=True,
                    help='Specify the path to the directory to store predictions')
# model
parser.add_argument('--path_model', type=str, default='',
                    help='Specify the path to the trained model')
# data
parser.add_argument('--path_data', type=str, required=True, default='~/data/bxl',
                    help='Specify the path to the data directory where img/ labels/ (and bm/) directories can be found')
parser.add_argument('--test', action="store_true", default=False,
                    help="whether to use the test set or not. (default to validation set)")
# parallel computation
parser.add_argument('--num_workers', type=int, default=10, help='Number of workers to preprocess images')
# hyperparameters
parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold')

parser.add_argument('--compute_dice', action="store_true", default=False,
                    help="Whether to compute the dice over all the dataset after having predicted it")


def get_default_device():
    """ Set device """
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def main(args):
    os.makedirs(args.path_pred, exist_ok=True)
    device = get_default_device()
    torch.multiprocessing.set_sharing_strategy('file_system')

    '''' Initialise dataloaders '''
    val_loader = get_val_dataloader(data_dir=args.path_data,
                                    num_workers=args.num_workers,
                                    test=args.test,
                                    cache_rate=0)

    ''' Load trained model  '''
    in_channels = len(args.sequences)
    path_pred = os.path.join(args.path_pred, os.path.basename(os.path.dirname(args.path_model)))
    os.makedirs(path_pred, exist_ok=True)

    model = UNet3D(in_channels, num_classes=2)
    model.load_state_dict(torch.load(args.path_model))
    model.to(device)
    model.eval()

    act = torch.nn.Softmax(dim=1)
    th = args.threshold
    roi_size = (96, 96, 96)
    sw_batch_size = 4

    ''' Predictions loop '''
    with torch.no_grad():
        avg_dsc = 0
        avg_ndsc = 0
        n_subjects = 0
        for count, batch_data in tqdm(enumerate(val_loader)):
            inputs = batch_data["image"].to(device)
            foreground_mask = batch_data["brain_mask"].squeeze().cpu().numpy()

            # get ensemble predictions
            all_outputs = []
            outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, model, mode='gaussian', overlap=0.5)
            semantic_pred = outputs

            semantic_pred = act(semantic_pred).cpu().numpy()
            semantic_pred = np.squeeze(semantic_pred[0, 1])
            del inputs, outputs
            torch.cuda.empty_cache()

            # Apply brainmask
            semantic_pred *= foreground_mask

            # get image metadata
            meta_dict = args.sequences[0] + "_meta_dict"
            original_affine = batch_data[meta_dict]['original_affine'][0]
            affine = batch_data[meta_dict]['affine'][0]
            spatial_shape = batch_data[meta_dict]['spatial_shape'][0]
            filename_or_obj = batch_data[meta_dict]['filename_or_obj'][0]
            filename_or_obj = os.path.basename(filename_or_obj)

            # obtain and save prediction probability mask
            filename = filename_or_obj[:14] + "_pred_prob.nii.gz"
            filepath = os.path.join(path_pred, filename)
            write_nifti(semantic_pred, filepath,
                        affine=original_affine,
                        target_affine=affine,
                        output_spatial_shape=spatial_shape)

            # obtain and save binary segmentation mask
            seg = semantic_pred.copy()
            seg[seg >= th] = 1
            seg[seg < th] = 0
            seg = np.squeeze(seg)

            seg = remove_small_lesions_from_binary_segmentation(seg)

            filename = filename_or_obj[:14] + "_pred-binary.nii.gz"
            filepath = os.path.join(path_pred, filename)
            write_nifti(seg, filepath,
                        affine=original_affine,
                        target_affine=affine,
                        mode='nearest',
                        output_spatial_shape=spatial_shape)

            if args.compute_dice:
                if args.test:
                    gt = nib.load(os.path.join(args.path_data, 'test', 'labels',
                                               filename_or_obj[:14] + "_mask-classes.nii.gz")).get_fdata()
                    gt = (gt > 0).astype(np.uint8)
                else:
                    gt = np.squeeze(batch_data["label"].cpu().numpy())

                dsc = dice_metric(gt, seg)
                ndsc = dice_norm_metric(gt, seg)
                print(filename_or_obj[:14], "DSC:", round(dsc, 3), " /  nDSC:", round(ndsc, 3))
                avg_dsc += dsc
                avg_ndsc += ndsc
            n_subjects += 1
    avg_dsc /= n_subjects
    avg_ndsc /= n_subjects
    print(f"Average Dice score for this subset is {avg_dsc}")
    print(f"Average normalizd Dice score for this subset is {avg_ndsc}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
