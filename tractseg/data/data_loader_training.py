"""
Code to load data and to create batches of 2D slices from 3D images.

Info:
Dimensions order for DeepLearningBatchGenerator: (batch_size, channels, x, y, [z])
"""

from pathlib import Path
from os.path import join
import random

import multiprocessing
import numpy as np
import nibabel as nib
import nrrd

from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.noise_transforms import GaussianBlurTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.augmentations.crop_and_pad_augmentations import crop

import tractseg.config as config
from tractseg.data.custom_transformations import ResampleTransformLegacy
from tractseg.data.custom_transformations import FlipVectorAxisTransform

# from batchgenerators.transforms.sample_normalization_transforms import ZeroMeanUnitVarianceTransform
from tractseg.data.DLDABG_standalone import ZeroMeanUnitVarianceTransform as ZeroMeanUnitVarianceTransform_Standalone
from tractseg.data.spatial_transform_peaks import SpatialTransformPeaks
from tractseg.data.spatial_transform_custom import SpatialTransformCustom
from tractseg.libs import data_utils
from tractseg.libs import peak_utils
from tractseg.libs import exp_utils


def load_training_data(subject):
    """
    Load data and labels for one subject from the training set. Cut and scale to make them have
    correct size.

    Args:
        subject: subject id (string)

    Returns:
        data and labels as 3D array
    """

    def load_img(path_img):
        if path_img.suffixes == [".nrrd"]:
            if config.SEG_INPUT == "fodfs":
                # bonndit output: (16, x, y, z)
                data_img, _ = nrrd.read(path_img)
                data_img = data_img[1:].transpose(1, 2, 3, 0)
            else:
                # bonndit output: (4, r, x, y, z)
                # TODO: Do this as preprocessing?
                data_img, _ = nrrd.read(path_img)
                # TODO: Use lambda?
                data_img = data_img[1:].transpose(2, 3, 4, 1, 0)
                data_img = data_img.reshape(*data_img.shape[:-2], -1)
        elif path_img.suffixes == [".nii", ".gz"]:
            # TODO: try np.asarray(array_img.dataobj)
            data_img = nib.load(path_img).get_fdata()
        else:
            raise ValueError("Unsupported input file type.")

        return data_img

    path_subject = Path(config.PATH_DATA) / subject
    # print(subject)
    data = load_img(path_subject / config.DIR_FEATURES / config.FILENAME_FEATURES)
    seg = load_img(path_subject / config.DIR_LABELS / config.FILENAME_LABELS)
    # print(f"{subject} success")
    return data, seg


class BatchGenerator2D_Nifti_random(SlimDataLoaderBase):
    """
    Randomly selects subjects and slices and creates batch of 2D slices.

    Takes image IDs provided via self._data, randomly selects one ID,
    loads the nifti image and randomly samples 2D slices from it.

    Timing:
    About 2s per 54-batch 45 bundles 1.25mm.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_train_batch(self):
        subjects = self._data[0]
        subject_idx = int(random.uniform(0, len(subjects)))

        data, seg = load_training_data(subjects[subject_idx])

        if config.NR_OF_GRADIENTS == 18 * config.NR_SLICES:
            data = peak_utils.peaks_to_tensors(data)

        slice_direction = data_utils.slice_dir_to_int(config.TRAINING_SLICE_DIRECTION)
        if data.shape[slice_direction] <= self.batch_size:
            print("INFO: Batch size bigger than nr of slices. Therefore sampling with replacement.")
            slice_idxs = np.random.choice(data.shape[slice_direction], self.batch_size, True, None)
        else:
            slice_idxs = np.random.choice(data.shape[slice_direction], self.batch_size, False, None)

        if config.NR_SLICES > 1:
            x, y = data_utils.sample_Xslices(
                data,
                seg,
                slice_idxs,
                slice_direction=slice_direction,
                labels_type=exp_utils.get_type_labels(config.TYPE_LABELS),
                slice_window=config.NR_SLICES,
            )
        else:
            x, y = data_utils.sample_slices(data, seg, slice_idxs, slice_direction=slice_direction)

        if config.PAD_TO_SQUARE:
            x, y = crop(x, y, crop_size=tuple(config.SHAPE_INPUT))
        else:
            # Works -> results as good?
            # Will pad each axis to be multiple of 16. (Each sample can end up having different dimensions. Also x and y can be different)
            # This is needed for the Schizo dataset.
            x = pad_nd_image(x, shape_must_be_divisible_by=(16, 16), mode="constant", kwargs={"constant_values": 0})
            y = pad_nd_image(y, shape_must_be_divisible_by=(16, 16), mode="constant", kwargs={"constant_values": 0})

        # Does not make it slower
        x = np.nan_to_num(x).astype(np.float32)
        y = np.nan_to_num(y).astype(np.float32)

        # (batch_size, channels, x, y, [z])
        data_dict = {"data": x, "seg": y, "slice_dir": slice_direction}
        return data_dict


class BatchGenerator2D_Npy_random(SlimDataLoaderBase):
    """
    Takes image ID provided via self._data, loads the Npy (numpy array) image and randomly samples 2D slices from it.
    Needed for fusion training.

    Timing:
    About 2s per 54-batch 45 bundles 1.25mm.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_train_batch(self):
        subjects = self._data[0]
        subject_idx = int(random.uniform(0, len(subjects)))

        if config.TYPE == "combined":
            if np.random.random() < 0.5:
                data = np.load(join(config.PATH_DATA, "HCP_fusion_npy_270g_125mm", subjects[subject_idx], "270g_125mm_xyz.npy"), mmap_mode="r")
            else:
                data = np.load(join(config.PATH_DATA, "HCP_fusion_npy_32g_25mm", subjects[subject_idx], "32g_25mm_xyz.npy"), mmap_mode="r")
            data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], data.shape[3] * data.shape[4]))
            seg = np.load(join(config.PATH_DATA, subjects[subject_idx], config.FILENAME_LABELS + ".npy"), mmap_mode="r")
        else:
            data = np.load(join(config.PATH_DATA, subjects[subject_idx], config.FILENAME_FEATURES + ".npy"), mmap_mode="r")
            seg = np.load(join(config.PATH_DATA, subjects[subject_idx], config.FILENAME_LABELS + ".npy"), mmap_mode="r")

        data = np.nan_to_num(data)
        seg = np.nan_to_num(seg)

        slice_idxs = np.random.choice(data.shape[0], self.batch_size, False, None)
        slice_direction = data_utils.slice_dir_to_int(config.TRAINING_SLICE_DIRECTION)
        x, y = data_utils.sample_slices(data, seg, slice_idxs, slice_direction=slice_direction)

        # (batch_size, channels, x, y, [z])
        data_dict = {"data": x, "seg": y}
        return data_dict


class DataLoaderTraining:
    def _augment_data(self, batch_generator, type=None):
        tfs = []

        if config.NORMALIZE_DATA:
            # TODO: Use original transform as soon as bug fixed in batchgenerators
            # tfs.append(ZeroMeanUnitVarianceTransform(per_channel=config.NORMALIZE_PER_CHANNEL))
            tfs.append(ZeroMeanUnitVarianceTransform_Standalone(per_channel=config.NORMALIZE_PER_CHANNEL))

        if config.SPATIAL_TRANSFORM == "SpatialTransformPeaks":
            SpatialTransformUsed = SpatialTransformPeaks
        elif config.SPATIAL_TRANSFORM == "SpatialTransformCustom":
            SpatialTransformUsed = SpatialTransformCustom
        else:
            SpatialTransformUsed = SpatialTransform

        if config.DATA_AUGMENTATION:
            if type == "train":
                # patch_center_dist_from_border:
                #   if 144/2=72 -> always exactly centered; otherwise a bit off center
                #   (brain can get off image and will be cut then)
                if config.DAUG_SCALE:
                    if config.INPUT_RESCALING:
                        source_mm = 2  # for bb
                        target_mm = float(config.RESOLUTION[:-2])
                        scale_factor = target_mm / source_mm
                        scale = (scale_factor, scale_factor)
                    else:
                        scale = (0.9, 1.5)

                    if config.PAD_TO_SQUARE:
                        patch_size = tuple(config.SHAPE_INPUT)
                    else:
                        patch_size = None  # keeps dimensions of the data

                    # spatial transform automatically crops/pads to correct size
                    center_dist_from_border = int(config.SHAPE_INPUT[0] / 2.0) - 10  # (144,144) -> 62
                    tfs.append(
                        SpatialTransformUsed(
                            patch_size,
                            patch_center_dist_from_border=center_dist_from_border,
                            do_elastic_deform=config.DAUG_ELASTIC_DEFORM,
                            alpha=tuple(config.DAUG_ALPHA),
                            sigma=tuple(config.DAUG_SIGMA),
                            do_rotation=config.DAUG_ROTATE,
                            angle_x=tuple(config.DAUG_ROTATE_ANGLE),
                            angle_y=tuple(config.DAUG_ROTATE_ANGLE),
                            angle_z=tuple(config.DAUG_ROTATE_ANGLE),
                            do_scale=True,
                            scale=scale,
                            border_mode_data="constant",
                            border_cval_data=0,
                            order_data=3,
                            border_mode_seg="constant",
                            border_cval_seg=0,
                            order_seg=0,
                            random_crop=True,
                            p_el_per_sample=config.P_SAMP,
                            p_rot_per_sample=config.P_SAMP,
                            p_scale_per_sample=config.P_SAMP,
                        )
                    )

                if config.DAUG_RESAMPLE:
                    tfs.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), p_per_sample=0.2, per_channel=False))

                if config.DAUG_RESAMPLE_LEGACY:
                    tfs.append(ResampleTransformLegacy(zoom_range=(0.5, 1)))

                if config.DAUG_GAUSSIAN_BLUR:
                    tfs.append(
                        GaussianBlurTransform(blur_sigma=tuple(config.DAUG_BLUR_SIGMA), different_sigma_per_channel=False, p_per_sample=config.P_SAMP)
                    )

                if config.DAUG_NOISE:
                    tfs.append(GaussianNoiseTransform(noise_variance=tuple(config.DAUG_NOISE_VARIANCE), p_per_sample=config.P_SAMP))

                if config.DAUG_MIRROR:
                    tfs.append(MirrorTransform())

                if config.DAUG_FLIP_PEAKS:
                    tfs.append(FlipVectorAxisTransform())

        tfs.append(NumpyToTensor(keys=["data", "seg"], cast_to="float"))

        # Set num_processes rather low because otherwise threads get killed for some reason (RAM looks actually okay, no idea why this happens)
        # This choice is based on the hard-coded value 15 processes that is used in the code as provided with the original paper by Wasserthal et al.
        # In the paper they state that the used a 48-core Intel Xeon CPU, so let's set this to one half of the core number.
        num_processes = multiprocessing.cpu_count() // 2
        # num_cached_per_queue 1 or 2 does not really make a difference
        batch_gen = MultiThreadedAugmenter(
            batch_generator, Compose(tfs), num_processes=num_processes, num_cached_per_queue=2, seeds=None, pin_memory=True
        )
        return batch_gen  # data: (batch_size, channels, x, y), seg: (batch_size, channels, x, y)

    def get_batch_generator(self, batch_size=128, type=None, subjects=None):
        data = subjects
        seg = []

        if config.TYPE == "combined":
            batch_gen = BatchGenerator2D_Npy_random((data, seg), batch_size=batch_size)
        else:
            batch_gen = BatchGenerator2D_Nifti_random((data, seg), batch_size=batch_size)

        batch_gen = self._augment_data(batch_gen, type=type)

        return batch_gen
