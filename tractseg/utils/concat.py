import argparse
import nibabel as nib
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Concatenate nifti images into single nifti image.")
    parser.add_argument("-i", dest="path_dir_input", required=True)
    parser.add_argument("-o", dest="path_output", required=True)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    imgs = [nib.load(path_img) for path_img in Path(args.path_dir_input).glob("*.nii.gz")]
    img_concatenated = nib.concat_images(imgs)
    nib.save(img_concatenated, args.path_output)


if __name__ == "__main__":
    main()
