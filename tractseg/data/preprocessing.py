from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import joblib
from joblib import Parallel
import nibabel as nib
import numpy as np
import tractseg.data.subjects
import tractseg.config as config
from tractseg.libs import data_utils
from tractseg.libs import exp_utils


def create_preprocessed_files(subject, overwrite=True):
    print(f"idx: {subject}")

    # Estimate bounding box from this file and then apply it to all other files
    bb_file = "12g_125mm_peaks"
    filenames_data = ["12g_125mm_raw32g", "270g_125mm_raw32g"]
    filenames_seg = []

    exp_utils.make_dir(config.PATH_DATA / f"{config.NAME_DATASET}_preprocessed" / subject)

    bb_file_path = config.PATH_NETWORK / config.NAME_DATASET / subject / bb_file + ".nii.gz"
    if not bb_file_path.exists():
        print("Missing file: {}-{}".format(subject, bb_file))
        raise IOError("File missing")

    # Get bounding box
    data = nib.load(bb_file_path).get_fdata()
    _, _, bbox, _ = data_utils.crop_to_nonzero(np.nan_to_num(data))

    for idx, filename in enumerate(filenames_data):
        path_src = config.PATH_NETWORK / config.NAME_DATASET / subject / filename + ".nii.gz"
        path_target = config.PATH_DATA / f"{config.NAME_DATASET}_preprocessed" / subject / filename + ".nii.gz"
        if path_target.exists() and not overwrite:
            print(f"Already done: {subject} - {filename}")
        elif path_src.exists():
            img = nib.load(path_src)
            data = img.get_fdata()
            affine = img.affine
            data = np.nan_to_num(data)

            # Add channel dimension if does not exist yet
            if len(data.shape) == 3:
                data = data[..., None]

            data, _, _, _ = data_utils.crop_to_nonzero(data, bbox=bbox)

            # np.save(join(C.PATH_DATA, DATASET_FOLDER_PREPROC, subject, filename + ".npy"), data)
            nib.save(nib.Nifti1Image(data, affine), path_target)
        else:
            raise IOError(f"Missing file: {subject}-{idx}".format(subject, idx))

    for idx, filename in enumerate(filenames_seg):
        path_src = config.PATH_NETWORK / config.NAME_DATASET / subject / filename + ".nii.gz"
        path_target = config.PATH_DATA / f"{config.NAME_DATASET}_preprocessed" / subject / filename + ".nii.gz"
        if path_target.exists() and not overwrite:
            print(f"Already done: {subject} - {filename}")
        elif path_src.exists():
            img = nib.load(path_src)
            data = img.get_fdata()
            data, _, _, _ = data_utils.crop_to_nonzero(data, bbox=bbox)
            # np.save(join(C.PATH_DATA, DATASET_FOLDER_PREPROC, subject, filename + ".npy"), data)
            nib.save(nib.Nifti1Image(data, img.affine), path_target)
        else:
            raise IOError(f"Missing seg file: {subject}-{idx}")


if __name__ == "__main__":
    """
    Run this script to crop images + segmentations to brain area. Then save as nifti.
    Reduces datasize and therefore IO by at least factor of 2.
    """
    print(f"Output folder: {config.DATASET_FOLDER_PREPROC}")
    subjects = tractseg.data.subjects.get_all_subjects(dataset=config.NAME_DATASET)
    Parallel(n_jobs=12)(joblib.delayed(create_preprocessed_files)(subject) for subject in subjects)
