import importlib_resources
from pathlib import Path

import argparse
import nibabel as nib
import numpy as np
import nrrd

import tractseg.config as config
from tractseg.data import dataset_specific_utils


def parse_args():
    parser = argparse.ArgumentParser(description="Run this script to crop nifti images to the brain area.")

    parser.add_argument("--config_exp", metavar="name", help="Name of experiment configuration to use", required=True)
    parser.add_argument("--ref", dest="path_rel_ref", help="Relative path to reference file", required=True)
    parser.add_argument("--spatial_channels_last", action="store_true")
    args = parser.parse_args()
    return args


def read_config_exp(filename_config=None):
    path_dir_configs_exp = importlib_resources.files("tractseg.experiments")
    config.set_config_exp(path_dir_configs_exp / "base.yaml")

    if filename_config is not None:
        config.set_config_exp(path_dir_configs_exp / "custom" / filename_config)


def get_bbox(data, value_background=0):
    coords = np.nonzero(np.asarray(data != value_background))
    bbox = np.array(
        [
            [np.min(coords[0]), np.max(coords[0])],
            [np.min(coords[1]), np.max(coords[1])],
            [np.min(coords[2]), np.max(coords[2])],
        ],
        dtype=int,
    )

    return bbox


def crop_to_bbox(img, bbox, spatial_channels_last=False):
    if spatial_channels_last:
        img_cropped = img[..., bbox[0][0] : bbox[0][1], bbox[1][0] : bbox[1][1], bbox[2][0] : bbox[2][1]]
    else:
        img_cropped = img[bbox[0][0] : bbox[0][1], bbox[1][0] : bbox[1][1], bbox[2][0] : bbox[2][1], ...]
    return img_cropped


def load_img(path):
    header = None
    affine = None
    if path.suffixes == [".nrrd"]:
        data, header = nrrd.read(path)
    elif path.suffixes == [".nii", ".gz"]:
        img = nib.load(path)
        data = img.get_fdata()
        affine = img.affine
    else:
        raise ValueError("Unsupported reference file type.")

    data = np.nan_to_num(data)
    return data, header, affine


def save_img(path, data, header, affine):
    if path.suffixes == [".nrrd"]:
        nrrd.write(path.as_posix(), data, header)
    elif path.suffixes == [".nii", ".gz"]:
        img_output = nib.Nifti1Image(data, affine)
        nib.save(img_output, path)
    else:
        raise ValueError("Unsupported reference file type.")


def get_largest_bbox(subjects, path_rel_ref, spatial_channels_last=False):
    bboxes = []
    for subject in subjects:
        path_ref = Path(config.PATH_DATA) / subject / path_rel_ref
        data_img, _, _ = load_img(path_ref)
        data_img = data_img[..., np.newaxis, :, :, :] if spatial_channels_last else data_img[:, :, :, np.newaxis, ...]
        bbox = get_bbox(data_img)
        bboxes += [bbox]

    bboxes = np.asarray(bboxes)
    bbox_union = np.array(
        [
            [np.min(bboxes[:, 0, 0]), np.max(bboxes[:, 0, 1])],
            [np.min(bboxes[:, 1, 0]), np.max(bboxes[:, 1, 1])],
            [np.min(bboxes[:, 2, 0]), np.max(bboxes[:, 2, 1])],
        ],
        dtype=int,
    )
    print(np.argmax(bboxes[:, 1, 1]))
    return bbox_union


def main():
    args = parse_args()
    read_config_exp(args.config_exp)

    subjects = dataset_specific_utils.get_subjects(config.DATASET)

    bbox_union = get_largest_bbox(subjects, args.path_rel_ref, args.spatial_channels_last)
    print("Bounding box:", bbox_union)

    for subject in subjects:
        print(f"Cropping features and tracts for subject {subject}.")
        path_features = Path(config.PATH_DATA) / subject / config.DIR_FEATURES / config.FILENAME_FEATURES
        path_labels = Path(config.PATH_DATA) / subject / config.DIR_LABELS / config.FILENAME_LABELS

        data, header, affine = load_img(path_features)
        data = crop_to_bbox(data, bbox_union, args.spatial_channels_last)
        path_cropped = path_features.with_stem(path_features.stem + "_cropped")
        save_img(path_cropped, data, header, affine)

        data, header, affine = load_img(path_labels)
        data = crop_to_bbox(data, bbox_union, args.spatial_channels_last)
        path_cropped = path_labels.with_name(path_labels.name.split(".")[0] + "_cropped" + "".join(path_labels.suffixes))
        save_img(path_cropped, data, header, affine)


if __name__ == "__main__":
    main()
