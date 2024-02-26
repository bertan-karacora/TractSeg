import multiprocessing
from pathlib import Path

import batchgenerators.augmentations as bgaug
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
from batchgenerators.transforms.noise_transforms import GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.sample_normalization_transforms import ZeroMeanUnitVarianceTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor
import numpy as np
import nibabel as nib
import nrrd

import tractseg.config as config
from tractseg.data.spatial_transform_peaks import SpatialTransformPeaks
from tractseg.data.spatial_transform_fodfs import SpatialTransformFodfs
from tractseg.data.noise_transform_fodfs import GaussianNoiseTransformFodfs
from tractseg.libs import data_utils
from tractseg.libs import exp_utils


def load_nrrd(path):
    img, _ = nrrd.read(path)
    return img


def load_nifti(path):
    img = nib.load(path).get_fdata()
    return img


def load_img(path):
    if path.suffixes == [".nrrd"]:
        img = load_nrrd(path)
    elif path.suffixes == [".nii", ".gz"]:
        img = load_nifti(path)
    else:
        raise ValueError("Unsupported input file type.")

    return img


def load_training_data(subject):
    path_subject = Path(config.PATH_DATA) / subject
    path_features = path_subject / config.DIR_FEATURES / config.FILENAME_FEATURES
    path_labels = path_subject / config.DIR_LABELS / config.FILENAME_LABELS

    features = load_img(path_features)
    labels = load_img(path_labels)

    return features, labels


def sample_slices(features, labels, batch_size, direction_slicing, num_slices):
    if features.shape[direction_slicing] <= batch_size:
        print("INFO: Batch size bigger than nr of slices. Therefore sampling with replacement.")
        ind_slices = np.random.choice(features.shape[direction_slicing], batch_size, True, None)
    else:
        ind_slices = np.random.choice(features.shape[direction_slicing], batch_size, False, None)

    if num_slices > 1:
        batch_features, batch_labels = data_utils.sample_Xslices(
            features,
            labels,
            ind_slices,
            slice_direction=direction_slicing,
            labels_type=exp_utils.get_type_labels(config.TYPE_LABELS),
            slice_window=num_slices,
        )
    else:
        batch_features, batch_labels = data_utils.sample_slices(
            features,
            labels,
            ind_slices,
            slice_direction=direction_slicing,
        )

    return batch_features, batch_labels


def reshape_batch(batch_features, batch_labels, shape_input):
    if config.PAD_TO_SQUARE:
        # TODO: What about pad_and_scale_img_to_square_img?
        batch_features, batch_labels = bgaug.crop_and_pad_augmentations.crop(
            batch_features,
            batch_labels,
            crop_size=shape_input,
        )
    else:
        # Will pad each axis to be multiple of 16. (Each sample can end up having different dimensions. Also x and y can be different)
        # This is needed for the Schizo dataset.
        batch_features = bgaug.utils.pad_nd_image(
            batch_features,
            shape_must_be_divisible_by=(16, 16),
            mode="constant",
            kwargs={"constant_values": 0},
        )
        batch_labels = bgaug.utils.pad_nd_image(
            batch_labels,
            shape_must_be_divisible_by=(16, 16),
            mode="constant",
            kwargs={"constant_values": 0},
        )

    return batch_features, batch_labels


class BatchGenerator(SlimDataLoaderBase):
    def __init__(self, subjects, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subjects = subjects

    def generate_train_batch(self):
        ind_subject = np.random.randint(len(self.subjects))
        features, labels = load_training_data(self.subjects[ind_subject])

        direction_slicing = data_utils.slice_dir_to_int(config.TRAINING_SLICE_DIRECTION)
        batch_features, batch_labels = sample_slices(features, labels, self.batch_size, direction_slicing, config.NUM_SLICES)

        batch_features, batch_labels = reshape_batch(batch_features, batch_labels, tuple(config.SHAPE_INPUT))

        batch_features = np.nan_to_num(batch_features).astype(np.float32)
        batch_labels = np.nan_to_num(batch_labels).astype(np.float32)

        data_dict = {
            "data": batch_features,
            "seg": batch_labels,
            "direction_slicing": direction_slicing,
        }
        return data_dict


class DataLoaderTraining:
    def _augment_data(self, batch_generator, type=None, num_threads=None):
        # Not beautiful but better than the originally in TractSeg.
        if config.SPATIAL_TRANSFORM == "SpatialTransformFodfs":
            SpatialTransformUsed = SpatialTransformFodfs
            NoiseTransformUsed = GaussianNoiseTransformFodfs
        elif config.SPATIAL_TRANSFORM == "SpatialTransformPeaks":
            SpatialTransformUsed = SpatialTransformPeaks
            NoiseTransformUsed = GaussianNoiseTransform
        else:
            SpatialTransformUsed = SpatialTransform
            NoiseTransformUsed = GaussianNoiseTransform

        tfs = []

        if config.NORMALIZE_SLICES:
            tfs.append(ZeroMeanUnitVarianceTransform(per_channel=config.NORMALIZE_PER_CHANNEL))

        if config.DATA_AUGMENTATION and type == "train":
            if config.DAUG_SCALE:
                dist = int(config.SHAPE_INPUT[0] / 2.0) - 10  # (144,144) -> 62
                tfs.append(
                    SpatialTransformUsed(
                        tuple(config.SHAPE_INPUT) if config.PAD_TO_SQUARE else None,
                        random_crop=True,
                        patch_center_dist_from_border=[dist, dist],
                        do_elastic_deform=config.DAUG_ELASTIC_DEFORM,
                        alpha=tuple(config.DAUG_ALPHA),
                        sigma=tuple(config.DAUG_SIGMA),
                        do_rotation=config.DAUG_ROTATE,
                        angle_x=tuple(config.DAUG_ROTATE_ANGLE),
                        angle_y=tuple(config.DAUG_ROTATE_ANGLE),
                        angle_z=tuple(config.DAUG_ROTATE_ANGLE),
                        do_scale=config.DAUG_SCALE,
                        scale=tuple(config.DAUG_RESCALING),
                        border_mode_data="constant",
                        border_cval_data=0,
                        order_data=3,
                        border_mode_seg="constant",
                        border_cval_seg=0,
                        order_seg=0,
                        p_el_per_sample=config.P_SAMP,
                        p_rot_per_sample=config.P_SAMP,
                        p_scale_per_sample=config.P_SAMP,
                    )
                )

            if config.DAUG_RESAMPLE:
                tfs.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), p_per_sample=0.2, per_channel=False))

            if config.DAUG_GAUSSIAN_BLUR:
                tfs.append(
                    GaussianBlurTransform(blur_sigma=tuple(config.DAUG_BLUR_SIGMA), different_sigma_per_channel=False, p_per_sample=config.P_SAMP)
                )

            if config.DAUG_NOISE:
                tfs.append(NoiseTransformUsed(noise_variance=tuple(config.DAUG_NOISE_VARIANCE), p_per_sample=config.P_SAMP))

        tfs.append(NumpyToTensor(keys=["data", "seg"], cast_to="float"))

        batch_gen = MultiThreadedAugmenter(
            batch_generator,
            Compose(tfs),
            num_processes=num_threads,
            num_cached_per_queue=2,
            seeds=None,
            pin_memory=True,
        )

        return batch_gen

    def get_batch_generator(self, subjects=None, batch_size=32, type=None):
        # In the original implementation of TractSeg provided by Wasserthal et al., a number of processes is hard-coded as 15.
        # In the paper, it is stated that an 48-core Intel Xeon CPU is used, so let's
        # set this rather low because otherwise threads get killed as we easily run out of RAM.
        # From experiments, setting this to the number of physical (not logical) cores shows the best speed for low-core CPUs while being reasonable stable.
        num_threads = multiprocessing.cpu_count() // 2

        features, labels = [], []
        batch_gen = BatchGenerator(subjects, (features, labels), batch_size=batch_size, number_of_threads_in_multithreaded=num_threads)
        batch_gen = self._augment_data(batch_gen, type=type, num_threads=num_threads)

        return batch_gen
