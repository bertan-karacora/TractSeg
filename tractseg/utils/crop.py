import argparse
import nibabel as nib
import numpy as np
import nrrd


def parse_args():
    parser = argparse.ArgumentParser(
        description="""
            Run this script to crop nifti images to the brain area.
            Reduces datasize and therefore IO by at least factor of 2.
        """
    )

    parser.add_argument("-i", dest="path_input", required=True)
    parser.add_argument("-o", dest="path_output", required=True)
    parser.add_argument("--ref", dest="path_reference", required=True)
    parser.add_argument("--spatial_channels_last", action="store_true", default=False)
    args = parser.parse_args()
    return args


def bounding_box(data, value_background=0):
    coords = np.nonzero(np.asarray(data != value_background))
    bbox = np.array(
        [
            [np.min(coords[0]), np.max(coords[0]) + 1],
            [np.min(coords[1]), np.max(coords[1]) + 1],
            [np.min(coords[2]), np.max(coords[2]) + 1],
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


def main():
    args = parse_args()

    img = None
    if args.path_input.endswith(".nrrd"):
        data_img, img_header = nrrd.read(args.path_input)
    elif args.path_input.endswith(".nii.gz"):
        img = nib.load(args.path_input)
        data_img = np.nan_to_num(img.get_fdata())
    else:
        raise ValueError("Unsupported input file type.")

    img_reference = nib.load(args.path_reference)
    data_reference = np.nan_to_num(img_reference.get_fdata())

    bbox = bounding_box(data_reference)
    data_img = crop_to_bbox(data_img, bbox, args.spatial_channels_last)

    if args.path_output.endswith(".nrrd"):
        nrrd.write(args.path_output, data_img, img_header)
    elif args.path_output.endswith(".nii.gz"):
        img_output = nib.Nifti1Image(data_img, img.affine if img is not None else np.eye(4))
        nib.save(img_output, args.path_output)
    else:
        raise ValueError("Unsupported input file type.")


if __name__ == "__main__":
    main()
