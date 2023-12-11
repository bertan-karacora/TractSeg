import argparse
import dipy.tracking.utils as tracking_utils
import logging
import nibabel as nib
import numpy as np
import scipy as sp

# Set formatting of output
logging.basicConfig(format="%(levelname)s: %(message)s")
logging.getLogger().setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        description="""
            Convert a trk streamline file to a binary map.
            Example usage: python trk_2_binary.py -i CST_right.trk -o CST_right.nii.gz -ref nodif_brain_mask.nii.gz
        """
    )

    parser.add_argument("-i", metavar="filepath", dest="input", required=True)
    parser.add_argument("-o", metavar="filepath", dest="output", required=True)
    parser.add_argument("--ref", metavar="filepath", dest="reference", required=True)
    parser.add_argument("--legacy_format", action="store_true", help="Use for zenodo dataset v1.1.0 and below", default=False)
    args = parser.parse_args()
    return args


def get_number_of_points(streamlines):
    count = 0
    for sl in streamlines:
        count += len(sl)
    return count


def remove_small_blobs(img, threshold=1):
    """
    Find blobs/clusters of same label. Only keep blobs with more than threshold elements.
    This can be used for postprocessing.
    """
    # If using structure=np.ones((3, 3, 3): Also considers diagonal elements for determining if a element belongs to a blob
    # -> not good, because leaves hardly any small blobs we can remove.
    mask, number_of_blobs = sp.ndimage.label(img)
    counts = np.bincount(mask.flatten())

    logging.debug(f"Number of blobs before filtering: {number_of_blobs}")
    logging.debug(f"Pixel counts: {counts}")

    remove = np.nonzero(counts <= threshold)[0]
    for idx in remove:
        mask[mask == idx] = 0
    mask[mask > 0] = 1

    mask_after, number_of_blobs_after = sp.ndimage.label(mask)
    logging.debug(f"Number of blobs after filtering: {number_of_blobs_after}")
    return mask


def main():
    args = parse_args()

    # Read files.
    if args.legacy_format:
        streams, hdr = nib.trackvis.read(args.input)
        streamlines = [s[0] for s in streams]
    else:
        file_sl = nib.streamlines.load(args.input)
        streamlines = file_sl.streamlines
    img_ref = nib.load(args.reference)

    # Upsample Streamlines (very important, especially when using DensityMap Threshold, without upsampling eroded results).
    len_max_seq = abs(img_ref.affine[0, 0] / 4)
    streamlines = list(tracking_utils.subsegment(streamlines, len_max_seq))

    # Remember: Does not count if a fibers has no node inside of a voxel -> upsampling helps, but not perfect.
    # Counts the number of unique streamlines that pass through each voxel -> oversampling does not distort result.
    dm = tracking_utils.density_map(streamlines, affine=img_ref.affine, vol_dims=img_ref.get_fdata().shape)

    # Create Binary Map
    # Using higher Threshold problematic, because tends to remove valid parts (sparse fibers).
    dm_binary = dm > 0

    # Filter Blobs (might remove valid parts) -> do not use
    # dm_binary_c = dm_binary
    # dm_binary_c = remove_small_blobs(dm_binary_c, threshold=10)

    # Closing of Holes (not ideal because tends to remove valid holes, e.g. in MCP) -> do not use
    # size = 1
    # dm_binary_c = ndimage.binary_closing(dm_binary_c, structure=np.ones((size, size, size))).astype(dm_binary.dtype)

    # Save Binary Mask
    img_dm_binary = nib.Nifti1Image(dm_binary.astype(np.uint8), img_ref.affine)
    nib.save(img_dm_binary, args.output)


if __name__ == "__main__":
    main()
