import argparse
import dipy.tracking.utils as tracking_utils
import nibabel as nib
import numpy as np
import scipy as sp


def parse_args():
    parser = argparse.ArgumentParser(
        description="""
            Convert a trk streamline file to a binary map.
            Example usage: python trk_2_binary.py -i CST_right.trk -o CST_right.nii.gz -ref nodif_brain_mask.nii.gz
        """
    )
    parser.add_argument("-i", dest="path_input", required=True)
    parser.add_argument("-o", dest="path_output", required=True)
    parser.add_argument("--ref", dest="path_reference", required=True)
    # Not recommended (might remove valid parts).
    parser.add_argument("--remove_blobs", action="store_true", default=False)
    # Not recommended (not ideal because it tends to remove valid holes, e.g., in MCP).
    parser.add_argument("--close_holes", action="store_true", default=False)
    parser.add_argument("--subsegment_factor", type=float, default=0.25)

    args = parser.parse_args()
    return args


def remove_small_blobs(img, threshold=1):
    """
    Find blobs/clusters of same label. Only keep blobs with more than threshold elements.
    This can be used for postprocessing.
    """
    labels, num_labels = sp.ndimage.label(img)
    counts_labels = np.bincount(labels.flatten())

    labels_to_remove = np.nonzero(counts_labels <= threshold)[0]
    for label_to_remove in labels_to_remove:
        img[labels == label_to_remove] = 0

    return img


def main():
    args = parse_args()

    file_tracts = nib.streamlines.load(args.path_input)
    img_ref = nib.load(args.path_reference)

    # Upsample Streamlines
    # Very important, especially when using DensityMap Threshold.
    zooms = np.asarray(img_ref.header.get_zooms())
    len_segment_max = np.min(zooms) * args.subsegment_factor
    streamlines = list(tracking_utils.subsegment(file_tracts.streamlines, len_segment_max))

    # Does not count if a fibers has no node inside of a voxel, upsampling helps, but is not perfect.
    # Counts the number of unique streamlines that pass through each voxel, so oversampling does not distort the result.
    dm = tracking_utils.density_map(streamlines, affine=img_ref.affine, vol_dims=img_ref.shape)

    # Using higher threshold problematic, because tends to remove valid parts (sparse fibers).
    threshold = 1
    dm_bin = dm >= threshold

    if args.remove_blobs:
        dm_bin = remove_small_blobs(dm_bin, threshold=10)

    if args.close_holes:
        dm_bin = sp.ndimage.binary_closing(dm_bin, structure=np.ones((1, 1, 1)))

    # Nibabel does not support bool dtype.
    img_dm_binary = nib.Nifti1Image(dm_bin.astype(np.uint8), img_ref.affine)
    nib.save(img_dm_binary, args.path_output)


if __name__ == "__main__":
    main()
