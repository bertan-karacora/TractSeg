import os
from os.path import join

import numpy as np


MAPPING_TYPE_LABELS = {
    "int": np.int16,
    "float": np.float32,
}


def load_weights(model, path):
    print(f"Loading weights from {path}.")
    model.load_model(path)

    return model


def mkdir_exp(path, use_existing=False):
    """
    Create a new experiment folder. If it already exists, create new one with increasing number at the end.
    If not training model (only predicting): Use existing folder
    """

    if use_existing:
        if not path.is_dir():
            raise ValueError(f"{path} directory does not exist.")
    else:
        if path.is_dir():
            name_exp = path.name
            counter = 1
            while path.is_dir():
                counter += 1
                path = path.parent / f"{name_exp}-{counter}"

        path.mkdir(parents=True)

    return path


def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_best_weights_path(path_exp):
    path = list(path_exp.glob("best_weights_ep*.npz"))[0]
    return path


def get_bvals_bvecs_path(args):
    input_file_without_ending = os.path.basename(args.input).split(".")[0]
    if args.bvals:
        bvals = args.bvals
    else:
        bvals = join(os.path.dirname(args.input), input_file_without_ending + ".bvals")
    if args.bvecs:
        bvecs = args.bvecs
    else:
        bvecs = join(os.path.dirname(args.input), input_file_without_ending + ".bvecs")
    return bvals, bvecs


def get_brain_mask_path(predict_img_output, brain_mask, input):
    if brain_mask:
        return brain_mask

    brain_mask_path = join(predict_img_output, "nodif_brain_mask.nii.gz")
    if os.path.isfile(brain_mask_path):
        return brain_mask_path

    brain_mask_path = join(os.path.dirname(input), "nodif_brain_mask.nii.gz")
    if os.path.isfile(brain_mask_path):
        print("Loading brain mask from: {}".format(brain_mask_path))
        return brain_mask_path

    return None


def add_background_class(data):
    """
    Calculate BG class (where no other class is 1) and add it at idx=0 to array.

    Args:
        data: 3D array with bundle masks (nr_bundles, x,y,z)

    Returns:
        (x,y,z,nr_bundles+1)
    """
    s = data[0].shape
    mask_ml = np.zeros((s[0], s[1], s[2], len(data) + 1))
    background = np.ones((s[0], s[1], s[2]))  # everything that contains no bundle

    for idx in range(len(data)):
        mask = data[idx]
        mask_ml[:, :, :, idx + 1] = mask
        background[mask == 1] = 0  # remove this bundle from background

    mask_ml[:, :, :, 0] = background
    return mask_ml


def print_and_save(exp_path, text, only_log=False):
    if not only_log:
        print(text)
    try:
        with open(join(exp_path, "Log.txt"), "a") as f:  # a for append
            f.write(text)
            f.write("\n")
    except IOError:
        print("WARNING: Could not write to Log.txt file")


def print_verbose(verbose, text):
    if verbose:
        print(text)


def get_type_labels(str_type):
    if str_type in MAPPING_TYPE_LABELS:
        type_labels = MAPPING_TYPE_LABELS[str_type]
    else:
        raise ValueError(f"ERROR: Type of labels not recognized: {str_type}")
    return type_labels


def get_manual_exp_name_peaks(manual_exp_name, part):
    """
    If want to use manual experiment name for peak regression, replace part nr by X:
    e.g. PeaksPartX_HR_DAug_fold2
    -> will find correct part then automatically
    """
    if "PeaksPartX" in manual_exp_name:
        manual_exp_name_parts = manual_exp_name.split("X")
        return manual_exp_name_parts[0] + part[-1] + manual_exp_name_parts[1]
    else:
        return manual_exp_name
