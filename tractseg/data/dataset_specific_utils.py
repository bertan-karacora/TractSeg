import numpy as np

import tractseg.config as config
from tractseg.data import labels, subjects
from tractseg.libs import img_utils


def get_labels(labelset):
    l = labels.get_labels(labelset)
    return l


def get_subjects(dataset):
    s = subjects.get_subjects(dataset)
    return s


def get_cv_fold(fold):
    # 5 fold CV ok (score only 1%-point worse than 10 folds (80 vs 60 train subjects) (10 Fold CV impractical!)
    num_folds = 5
    num_folds_train, num_folds_val, num_folds_test = 3, 1, 1

    train = [(fold + i) % num_folds for i in range(num_folds_train)]
    validate = [(fold + num_folds_train + i) % num_folds for i in range(num_folds_val)]
    test = [(fold + num_folds_train + num_folds_val + i) % num_folds for i in range(num_folds_test)]

    s = subjects.get_subjects(config.DATASET)
    size_chunk = 21
    chunks_s = np.asarray([s[i : i + size_chunk] for i in range(0, len(s), size_chunk)])

    s_train = list(chunks_s[train].flatten())
    s_val = list(chunks_s[validate].flatten())
    s_test = list(chunks_s[test].flatten())
    return s_train, s_val, s_test


def scale_input_to_unet_shape(img4d):
    # HCP 1.25mm
    # (145,174,145)
    return img4d[1:, 15:159, 1:]  # (144,144,144)


def scale_input_to_original_shape(img4d):
    # HCP 1.25mm
    # (144,144,144)
    img_padded = img_utils.pad_4d_image_left(img4d, np.array([1, 15, 1, 0]), [146, 174, 146, img4d.shape[3]], pad_value=0)
    img_padded = img_padded[:-1, :, :-1, :]  # (145, 174, 145, none)
    return img_padded


def get_optimal_orientation_for_bundle(bundle):
    """
    Get optimal orientation if you want to plot the respective bundle.
    """
    bundles_orientation = {
        "AF_left": "sagittal",
        "AF_right": "sagittal",
        "ATR_left": "sagittal",
        "ATR_right": "sagittal",
        "CA": "coronal",
        "CC_1": "axial",
        "CC_2": "axial",
        "CC_3": "coronal",
        "CC_4": "coronal",
        "CC_5": "coronal",
        "CC_6": "coronal",
        "CC_7": "axial",
        "CG_left": "sagittal",
        "CG_right": "sagittal",
        "CST_left": "coronal",
        "CST_right": "coronal",
        "MLF_left": "sagittal",
        "MLF_right": "sagittal",
        "FPT_left": "sagittal",
        "FPT_right": "sagittal",
        "FX_left": "sagittal",
        "FX_right": "sagittal",
        "ICP_left": "sagittal",
        "ICP_right": "sagittal",
        "IFO_left": "sagittal",
        "IFO_right": "sagittal",
        "ILF_left": "sagittal",
        "ILF_right": "sagittal",
        "MCP": "axial",
        "OR_left": "axial",
        "OR_right": "axial",
        "POPT_left": "sagittal",
        "POPT_right": "sagittal",
        "SCP_left": "sagittal",
        "SCP_right": "sagittal",
        "SLF_I_left": "sagittal",
        "SLF_I_right": "sagittal",
        "SLF_II_left": "sagittal",
        "SLF_II_right": "sagittal",
        "SLF_III_left": "sagittal",
        "SLF_III_right": "sagittal",
        "STR_left": "sagittal",
        "STR_right": "sagittal",
        "UF_left": "sagittal",
        "UF_right": "sagittal",
        "CC": "sagittal",
        "T_PREF_left": "sagittal",
        "T_PREF_right": "sagittal",
        "T_PREM_left": "sagittal",
        "T_PREM_right": "sagittal",
        "T_PREC_left": "sagittal",
        "T_PREC_right": "sagittal",
        "T_POSTC_left": "sagittal",
        "T_POSTC_right": "sagittal",
        "T_PAR_left": "sagittal",
        "T_PAR_right": "sagittal",
        "T_OCC_left": "sagittal",
        "T_OCC_right": "sagittal",
        "ST_FO_left": "sagittal",
        "ST_FO_right": "sagittal",
        "ST_PREF_left": "sagittal",
        "ST_PREF_right": "sagittal",
        "ST_PREM_left": "sagittal",
        "ST_PREM_right": "sagittal",
        "ST_PREC_left": "sagittal",
        "ST_PREC_right": "sagittal",
        "ST_POSTC_left": "sagittal",
        "ST_POSTC_right": "sagittal",
        "ST_PAR_left": "sagittal",
        "ST_PAR_right": "sagittal",
        "ST_OCC_left": "sagittal",
        "ST_OCC_right": "sagittal",
    }

    return bundles_orientation[bundle]
