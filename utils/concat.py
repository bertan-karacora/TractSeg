import argparse
import nibabel as nib
from pathlib import Path

import tractseg.config as config
from tractseg.data import dataset_specific_utils


def parse_args():
    parser = argparse.ArgumentParser(description="Concatenate nifti images into single nifti image.")
    parser.add_argument("-i", dest="path_dir_input", required=True)
    parser.add_argument("-o", dest="path_output", required=True)
    parser.add_argument("--path_config_exp", metavar="path", help="Path of experiment configuration to use", required=True)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config.set_config_exp(Path(args.path_config_exp))

    # Get the correct order
    classes = dataset_specific_utils.get_classes(config.CLASSSET)

    path_dir_input = Path(args.path_dir_input)
    imgs = [nib.load(path_dir_input / f"{c}.nii.gz") for c in classes]
    img_concatenated = nib.concat_images(imgs)
    nib.save(img_concatenated, args.path_output)


if __name__ == "__main__":
    main()
