from pathlib import Path

import argparse
import nibabel as nib
import nrrd
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="""Run this script to preprocess feature 3D images.""")

    parser.add_argument("-i", dest="path_input", required=True)
    parser.add_argument("-o", dest="path_output", required=True)
    parser.add_argument("-t", dest="path_t1", required=False)
    parser.add_argument("-m", dest="path_5tt", required=False)
    parser.add_argument("--type", choices=["peaks", "fodfs", "rank_k", "combined_t1_wmmask", "peaks_tensor"], default="fodfs")
    args = parser.parse_args()
    return args


def load_nrrd(path):
    img, header_img = nrrd.read(path)
    return img, header_img


def load_nifti(path):
    img_nifti = nib.load(path)
    img = img_nifti.get_fdata()
    affine_img = img_nifti.affine
    return img, affine_img


def load_img(path):
    header_img, affine_img = None, None
    if path.suffixes == [".nrrd"]:
        img, header_img = load_nrrd(path)
    elif path.suffixes == [".nii", ".gz"]:
        img, affine_img = load_nifti(path)
    else:
        raise ValueError("Unsupported input file type.")

    return img, header_img, affine_img


def save_img(path, img, img_header=None, affine=None):
    if path.suffixes == [".nrrd"]:
        nrrd.write(str(path), img, img_header)
    elif path.suffixes == [".nii", ".gz"]:
        img_output = nib.Nifti1Image(img, affine if affine is not None else np.eye(4))
        nib.save(img_output, str(path))
    else:
        raise ValueError("Unsupported file type.")


def reorder_channels_fodf(img):
    # Bonndit output: (16, x, y, z)
    img = img[1:].transpose(1, 2, 3, 0)
    return img


def reorder_channels_rank_k(img):
    # Bonndit output: (4, r, x, y, z)
    # Low-rank approximation from bonndit is always from 4th-order tensors.
    order = 4
    np.power(img[0], 1.0 / order)

    img = img[1:] * np.power(img[0], 1.0 / order)
    img = img.transpose(2, 3, 4, 1, 0)
    img = img.reshape(*img.shape[:-2], -1)
    return img


def concatenate(*imgs):
    img = imgs[0]
    # Unfortunately need a for loop because numpy does not allow for concatenating variable lengths in a single statement.
    for img_c in imgs[1:]:
        img = np.append(img, img_c, axis=-1)
    return img


def peaks2tensor(img):
    peaks1, peaks2, peaks3 = img[..., :3], img[..., 3:6], img[..., 6:9]
    tensors = peaks1[..., None] * peaks1[..., None, :] + peaks2[..., None] * peaks2[..., None, :] + peaks3[..., None] * peaks3[..., None, :]
    tensors = tensors.reshape(*tensors.shape[:-2], -1)
    img = tensors[..., [0, 1, 2, 4, 5, 8]]
    return img


def main():
    args = parse_args()

    path_img = Path(args.path_input)
    img, header_img, affine_img = load_img(path_img)

    if args.type == "fodfs":
        img = reorder_channels_fodf(img)

    if args.type == "rank_k":
        img = reorder_channels_rank_k(img)

    if args.type == "combined_t1_wmmask":
        img = reorder_channels_fodf(img)

        path_t1 = Path(args.path_t1)
        img_t1, _, _ = load_img(path_t1)

        path_5tt = Path(args.path_5tt)
        img_5tt, _, _ = load_img(path_5tt)

        # Channel 2 contains white matter partial volume.
        img = concatenate(img, img_t1[..., None], img_5tt[..., 2, None])

    if args.type == "peaks_tensor":
        img = peaks2tensor(img)

    path_out = Path(args.path_output)
    save_img(path_out, img, header_img, affine_img)


if __name__ == "__main__":
    main()
