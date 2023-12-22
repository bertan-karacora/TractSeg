from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
from builtins import object
import numpy as np

import tractseg.config as config
from tractseg.libs import exp_utils
from tractseg.libs import data_utils
from tractseg.libs import peak_utils
from tractseg.data.DLDABG_standalone import ZeroMeanUnitVarianceTransform as ZeroMeanUnitVarianceTransform_Standalone
from tractseg.data.DLDABG_standalone import SingleThreadedAugmenter
from tractseg.data.DLDABG_standalone import Compose
from tractseg.data.DLDABG_standalone import NumpyToTensor

np.random.seed(1337)


class BatchGenerator2D_data_ordered_standalone(object):
    """
    Creates batch of 2D slices from one subject.

    Does not depend on DKFZ/BatchGenerators package. Therefore good for inference on windows
    where DKFZ/Batchgenerators do not work (because of MultiThreading problems)
    """

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

        if config.NR_SLICES > 1:
            x, y = data_utils.sample_Xslices(
                data, seg, slice_idxs, slice_direction=slice_direction, labels_type=exp_utils.get_correct_labels_type(), slice_window=config.NR_SLICES
            )
        else:
            x, y = data_utils.sample_slices(data, seg, slice_idxs, slice_direction=slice_direction, labels_type=exp_utils.get_correct_labels_type())

        data_dict = {"data": x, "seg": y}  # (batch_size, channels, x, y, [z])  # (batch_size, channels, x, y, [z])
        self.global_idx = new_global_idx
        return data_dict


class BatchGenerator3D_data_ordered_standalone(object):
    def __init__(self, data, batch_size=1):
        if batch_size != 1:
            raise ValueError("only batch_size=1 allowed")
        self.batch_size = batch_size
        self.global_idx = 0
        self._data = data

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate_train_batch()

    def generate_train_batch(self):
        data = self._data[0]  # (x, y, z, channels)
        seg = self._data[1]

        # Stop iterating if we reached end of data
        if self.global_idx >= 1:
            self.global_idx = 0
            raise StopIteration
        self.global_idx += self.batch_size

        x = data.transpose(3, 0, 1, 2)[np.newaxis, ...]  # channels have to be first, add batch_size of 1
        y = seg.transpose(3, 0, 1, 2)[np.newaxis, ...]

        data_dict = {"data": np.array(x), "seg": np.array(y)}  # (batch_size, channels, x, y, [z])  # (batch_size, channels, x, y, [z])
        return data_dict


class DataLoaderInference:
    """
    Data loader for only one subject and returning slices in ordered way.
    """

    def __init__(self, data=None, subject=None):
        """
        Set either data or subject, not both.

        Args:
            data: 4D numpy array with subject data
            subject: ID for a subject from the training data (string)
        """
        self.data = data
        self.subject = subject

    def _augment_data(self, batch_generator, type=None):
        tfs = []

        if config.NORMALIZE_DATA:
            tfs.append(ZeroMeanUnitVarianceTransform_Standalone(per_channel=config.NORMALIZE_PER_CHANNEL))

        tfs.append(NumpyToTensor(keys=["data", "seg"], cast_to="float"))

        batch_gen = SingleThreadedAugmenter(batch_generator, Compose(tfs))
        return batch_gen

    def get_batch_generator(self, batch_size=1):
        if self.data is not None:
            exp_utils.print_verbose(config.VERBOSE, "Loading data from PREDICT_IMG input file")
            data = np.nan_to_num(self.data)
            # Use dummy mask in case we only want to predict on some data (where we do not have ground truth))
            seg = np.zeros((config.SHAPE_INPUT[0], config.SHAPE_INPUT[0], config.SHAPE_INPUT[0], config.NR_OF_CLASSES)).astype(
                exp_utils.get_correct_labels_type()
            )
        elif self.subject is not None:
            if config.TYPE == "combined":
                # Load from npy file for Fusion
                data = np.load(join(config.PATH_DATA, self.subject, config.FILENAME_FEATURES + ".npy"), mmap_mode="r")
                seg = np.load(join(config.PATH_DATA, self.subject, config.FILENAME_LABELS + ".npy"), mmap_mode="r")
                data = np.nan_to_num(data)
                seg = np.nan_to_num(seg)
                data = np.reshape(data, (data.shape[0], data.shape[1], data.shape[2], data.shape[3] * data.shape[4]))
            else:
                from tractseg.data.data_loader_training import load_training_data

                data, seg = load_training_data(self.subject)

                # Convert peaks to tensors if tensor model
                if config.NR_OF_GRADIENTS == 18 * config.NR_SLICES:
                    data = peak_utils.peaks_to_tensors(data)

                data, transformation = data_utils.pad_and_scale_img_to_square_img(data, target_size=config.SHAPE_INPUT[0], nr_cpus=1)
                seg, transformation = data_utils.pad_and_scale_img_to_square_img(seg, target_size=config.SHAPE_INPUT[0], nr_cpus=1)
        else:
            raise ValueError("Neither 'data' nor 'subject' set.")

        if len(config.SHAPE_INPUT):
            batch_gen = BatchGenerator2D_data_ordered_standalone((data, seg), batch_size=batch_size)
        else:
            batch_gen = BatchGenerator3D_data_ordered_standalone((data, seg), batch_size=batch_size)

        batch_gen = self._augment_data(batch_gen, type=type)
        return batch_gen
