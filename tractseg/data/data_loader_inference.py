from builtins import object
import numpy as np

import tractseg.config as config
from tractseg.libs import exp_utils
from tractseg.libs import data_utils
from batchgenerators.transforms.sample_normalization_transforms import ZeroMeanUnitVarianceTransform
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.transforms.abstract_transforms import Compose


class BatchGenerator2D_data_ordered_standalone(object):
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.global_idx = 0
        self._data = data

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate_train_batch()

    def generate_train_batch(self):
        data = self._data[0]
        seg = self._data[1]

        if config.SLICE_DIRECTION == "x":
            end = data.shape[0]
        elif config.SLICE_DIRECTION == "y":
            end = data.shape[1]
        elif config.SLICE_DIRECTION == "z":
            end = data.shape[2]

        # Stop iterating if we reached end of data
        if self.global_idx >= end:
            self.global_idx = 0
            raise StopIteration

        new_global_idx = self.global_idx + self.batch_size

        # If we reach end, make last batch smaller, so it fits exactly for rest
        if new_global_idx >= end:
            new_global_idx = end  # not end-1, because this goes into range, and there automatically -1

        slice_idxs = list(range(self.global_idx, new_global_idx))
        slice_direction = data_utils.slice_dir_to_int(config.SLICE_DIRECTION)

        if config.NUM_SLICES > 1:
            x, y = data_utils.sample_Xslices(
                data,
                seg,
                slice_idxs,
                slice_direction=slice_direction,
                labels_type=exp_utils.get_type_labels(config.TYPE_LABELS),
                slice_window=config.NUM_SLICES,
            )
        else:
            x, y = data_utils.sample_slices(data, seg, slice_idxs, slice_direction=slice_direction)

        x = x.astype(np.float32)
        y = y.astype(np.float32)

        data_dict = {"data": x, "seg": y}
        self.global_idx = new_global_idx
        return data_dict


class DataLoaderInference:
    def __init__(self, data=None, subject=None):
        self.data = data
        self.subject = subject

        np.random.seed(1337)

    def _augment_data(self, batch_generator, type=None):
        tfs = []

        if config.NORMALIZE_SLICES:
            tfs.append(ZeroMeanUnitVarianceTransform(per_channel=config.NORMALIZE_PER_CHANNEL))

        tfs.append(NumpyToTensor(keys=["data", "seg"], cast_to="float"))

        batch_gen = SingleThreadedAugmenter(batch_generator, Compose(tfs))

        return batch_gen

    def get_batch_generator(self, batch_size=1):
        if self.data is not None:
            exp_utils.print_verbose(config.VERBOSE, "Loading data from PREDICT_IMG input file")
            data = np.nan_to_num(self.data)
            # Use dummy mask in case we only want to predict on some data (where we do not have ground truth))
            seg = np.zeros((config.SHAPE_INPUT[0], config.SHAPE_INPUT[0], config.SHAPE_INPUT[0], len(config.CLASSES))).astype(
                exp_utils.get_type_labels(config.TYPE_LABELS)
            )
        elif self.subject is not None:
            from tractseg.data.data_loader_training import load_training_data

            data, seg = load_training_data(self.subject)

            data, transformation = data_utils.pad_and_scale_img_to_square_img(data, target_size=config.SHAPE_INPUT[0], nr_cpus=1)
            seg, transformation = data_utils.pad_and_scale_img_to_square_img(seg, target_size=config.SHAPE_INPUT[0], nr_cpus=1)
        else:
            raise ValueError("Neither 'data' nor 'subject' set.")

        batch_gen = BatchGenerator2D_data_ordered_standalone((data, seg), batch_size=batch_size)

        batch_gen = self._augment_data(batch_gen, type=type)
        return batch_gen
