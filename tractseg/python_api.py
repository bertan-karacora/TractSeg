from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import importlib
import time
import os
from os.path import join
import numpy as np

import tractseg.config as config
from tractseg.libs import exp_utils
from tractseg.libs import utils
from tractseg.libs import data_utils
from tractseg.libs import direction_merger
from tractseg.libs import peak_utils
from tractseg.libs import img_utils
from tractseg.data.data_loader_inference import DataLoaderInference
from tractseg.data import dataset_specific_utils
from tractseg.libs import train
from tractseg.models.base_model import BaseModel

warnings.simplefilter("ignore", UserWarning)  # hide scipy warnings
warnings.simplefilter("ignore", FutureWarning)  # hide h5py warnings


def run_tractseg(
    data,
    output_type="tract_segmentation",
    single_orientation=False,
    dropout_sampling=False,
    threshold=0.5,
    bundle_specific_postprocessing=True,
    get_probs=False,
    peak_threshold=0.1,
    postprocess=False,
    peak_regression_part="All",
    input_type="peaks",
    blob_size_thr=50,
    nr_cpus=-1,
    verbose=False,
    manual_exp_name=None,
    inference_batch_size=1,
    tract_definition="TractQuerier+",
    bedpostX_input=False,
    tract_segmentations_path=None,
    TOM_dilation=1,
    unit_test=False,
):
    """
    Run TractSeg

    Args:
        data: input peaks (4D numpy array with shape [x,y,z,9])
        output_type: TractSeg can segment not only bundles, but also the end regions of bundles.
            Moreover it can create Tract Orientation Maps (TOM).
            'tract_segmentation' [DEFAULT]: Segmentation of bundles (72 bundles).
            'endings_segmentation': Segmentation of bundle end regions (72 bundles).
            'TOM': Tract Orientation Maps (20 bundles).
        single_orientation: Do not run model 3 times along x/y/z orientation with subsequent mean fusion.
        dropout_sampling: Create uncertainty map by monte carlo dropout (https://arxiv.org/abs/1506.02142)
        threshold: Threshold for converting probability map to binary map
        bundle_specific_postprocessing: Set threshold to lower and use hole closing for CA nd FX if incomplete
        get_probs: Output raw probability map instead of binary map
        peak_threshold: All peaks shorter than peak_threshold will be set to zero
        postprocess: Simple postprocessing of segmentations: Remove small blobs and fill holes
        peak_regression_part: Only relevant for output type 'TOM'. If set to 'All' (default) it will return all
            72 bundles. If set to 'Part1'-'Part4' it will only run for a subset of the bundles to reduce memory
            load.
        input_type: Always set to "peaks"
        blob_size_thr: If setting postprocess to True, all blobs having a smaller number of voxels than specified in
            this threshold will be removed.
        nr_cpus: Number of CPUs to use. -1 means all available CPUs.
        verbose: Show debugging infos
        manual_exp_name: Name of experiment if do not want to use pretrained model but your own one
        inference_batch_size: batch size (higher: a bit faster but needs more RAM)
        tract_definition: Select which tract definitions to use. 'TractQuerier+' defines tracts mainly by their
            cortical start and end region. 'xtract' defines tracts mainly by ROIs in white matter.
        bedpostX_input: Input peaks are generated by bedpostX
        tract_segmentations_path: path to the bundle_segmentations (only needed for peak regression to remove peaks
            outside of the segmentation mask)
        TOM_dilation: Dilation applied to the tract segmentations before using them to mask the TOMs.

    Returns:
        4D numpy array with the output of tractseg
        for tract_segmentation:     [x, y, z, nr_of_bundles]
        for endings_segmentation:   [x, y, z, 2*nr_of_bundles]
        for TOM:                    [x, y, z, 3*nr_of_bundles]
    """
    start_time = time.time()

    if manual_exp_name is None:
        path_config_exp = config.get_path_config_pretrained(
            input_type, output_type, dropout_sampling=dropout_sampling, tract_definition=tract_definition
        )
        config.set_config_exp(path_config_exp)
    else:
        config.set_config_exp(join(config.PATH_EXP, exp_utils.get_manual_exp_name_peaks(manual_exp_name, "Part1"), "config_exp.yaml"))

    # Do not do any postprocessing if returning probabilities (because postprocessing only works on binary)
    if get_probs:
        bundle_specific_postprocessing = False
        postprocess = False

    config.VERBOSE = verbose
    config.TRAIN = False
    config.TEST = False
    config.SEGMENT = False
    config.GET_PROBS = get_probs
    config.LOAD_WEIGHTS = True
    config.DROPOUT_SAMPLING = dropout_sampling
    config.THRESHOLD = threshold
    config.NUM_CPUS = nr_cpus
    config.SHAPE_INPUT = dataset_specific_utils.get_correct_input_dim()
    config.RESET_LAST_LAYER = False

    if config.TYPE_EXP == "tract_segmentation" and bundle_specific_postprocessing:
        config.GET_PROBS = True

    if manual_exp_name is not None and config.TYPE_EXP != "peak_regression":
        config.PATH_WEIGHTS = exp_utils.get_best_weights_path(join(config.EXP_PATH, manual_exp_name))
    else:
        if tract_definition == "TractQuerier+":
            if input_type == "peaks":
                if config.TYPE_EXP == "tract_segmentation" and config.DROPOUT_SAMPLING:
                    config.PATH_WEIGHTS = join(config.PATH_DIR_WEIGHTS, "pretrained_weights_tract_segmentation_v3.npz")
                elif config.TYPE_EXP == "tract_segmentation":
                    config.PATH_WEIGHTS = join(config.PATH_DIR_WEIGHTS, "pretrained_weights_tract_segmentation_v3.npz")
                elif config.TYPE_EXP == "endings_segmentation":
                    config.PATH_WEIGHTS = join(config.PATH_DIR_WEIGHTS, "pretrained_weights_endings_segmentation_v4.npz")
                elif config.TYPE_EXP == "dm_regression":
                    config.PATH_WEIGHTS = join(config.PATH_DIR_WEIGHTS, "pretrained_weights_dm_regression_v2.npz")
            else:  # T1
                if config.TYPE_EXP == "tract_segmentation":
                    config.PATH_WEIGHTS = join(
                        config.NETWORK_DRIVE, "hcp_exp_nodes/x_Pretrained_TractSeg_Models", "TractSeg_T1_125mm_DAugAll", "best_weights_ep142.npz"
                    )
                elif config.TYPE_EXP == "endings_segmentation":
                    config.PATH_WEIGHTS = join(config.PATH_DIR_WEIGHTS, "pretrained_weights_endings_segmentation_v1.npz")
                elif config.TYPE_EXP == "peak_regression":
                    config.PATH_WEIGHTS = join(config.PATH_DIR_WEIGHTS, "pretrained_weights_peak_regression_v1.npz")
        else:  # xtract
            if config.TYPE_EXP == "tract_segmentation":
                config.PATH_WEIGHTS = join(config.PATH_DIR_WEIGHTS, "pretrained_weights_tract_segmentation_xtract_v1.npz")
            elif config.TYPE_EXP == "dm_regression":
                config.PATH_WEIGHTS = join(config.PATH_DIR_WEIGHTS, "pretrained_weights_dm_regression_xtract_v1.npz")
            else:
                raise ValueError("bundle_definition xtract not supported in combination with this output type")

    if config.VERBOSE:
        print("Hyperparameters:")
        config.dump()

    data = np.nan_to_num(data)

    # runtime on HCP data: 0.9s
    data, seg_None, bbox, original_shape = data_utils.crop_to_nonzero(data)
    # runtime on HCP data: 0.5s
    data, transformation = data_utils.pad_and_scale_img_to_square_img(data, target_size=config.SHAPE_INPUT[0], nr_cpus=nr_cpus)

    if config.TYPE_EXP == "tract_segmentation" or config.TYPE_EXP == "endings_segmentation" or config.TYPE_EXP == "dm_regression":
        print("Loading weights from: {}".format(config.PATH_WEIGHTS))
        len(config.CLASSES) = len(dataset_specific_utils.get_classes(config.CLASSES)[1:])
        utils.download_pretrained_weights(
            experiment_type=config.TYPE_EXP, dropout_sampling=config.DROPOUT_SAMPLING, tract_definition=tract_definition
        )
        model = BaseModel(inference=True)
        if single_orientation:  # mainly needed for testing because of less RAM requirements
            data_loder_inference = DataLoaderInference(data=data)
            if config.DROPOUT_SAMPLING or config.TYPE_EXP == "dm_regression" or config.GET_PROBS:
                seg, _ = train.predict_img(
                    model,
                    data_loder_inference,
                    probs=True,
                    scale_to_world_shape=False,
                    only_prediction=True,
                    batch_size=inference_batch_size,
                    unit_test=unit_test,
                )
            else:
                seg, _ = train.predict_img(
                    model,
                    data_loder_inference,
                    probs=False,
                    scale_to_world_shape=False,
                    only_prediction=True,
                    batch_size=inference_batch_size,
                )
        else:
            seg_xyz, _ = direction_merger.get_seg_single_img_3_directions(
                model, data=data, scale_to_world_shape=False, only_prediction=True, batch_size=inference_batch_size
            )
            if config.DROPOUT_SAMPLING or config.TYPE_EXP == "dm_regression" or config.GET_PROBS:
                seg = direction_merger.mean_fusion(config.THRESHOLD, seg_xyz, probs=True)
            else:
                seg = direction_merger.mean_fusion(config.THRESHOLD, seg_xyz, probs=False)

    elif config.TYPE_EXP == "peak_regression":
        weights = {
            "Part1": "pretrained_weights_peak_regression_part1_v2.npz",
            "Part2": "pretrained_weights_peak_regression_part2_v2.npz",
            "Part3": "pretrained_weights_peak_regression_part3_v2.npz",
            "Part4": "pretrained_weights_peak_regression_part4_v2.npz",
        }
        if peak_regression_part == "All":
            parts = ["Part1", "Part2", "Part3", "Part4"]
            seg_all = np.zeros((data.shape[0], data.shape[1], data.shape[2], len(config.CLASSES) * 3))
        else:
            parts = [peak_regression_part]
            config.CLASSES = "All_" + peak_regression_part
            len(config.CLASSES) = 3 * len(dataset_specific_utils.get_classes(config.CLASSES)[1:])

        for idx, part in enumerate(parts):
            if manual_exp_name is not None:
                manual_exp_name_peaks = exp_utils.get_manual_exp_name_peaks(manual_exp_name, part)
                config.PATH_WEIGHTS = exp_utils.get_best_weights_path(join(config.EXP_PATH, manual_exp_name_peaks))
            else:
                config.PATH_WEIGHTS = join(config.PATH_DIR_WEIGHTS, weights[part])
            print("Loading weights from: {}".format(config.PATH_WEIGHTS))
            config.CLASSES = "All_" + part
            len(config.CLASSES) = 3 * len(dataset_specific_utils.get_classes(config.CLASSES)[1:])
            utils.download_pretrained_weights(
                experiment_type=config.TYPE_EXP, dropout_sampling=config.DROPOUT_SAMPLING, part=part, tract_definition=tract_definition
            )
            model = BaseModel(inference=True)

            if single_orientation:
                data_loder_inference = DataLoaderInference(data=data)
                seg, _ = train.predict_img(
                    model, data_loder_inference, probs=True, scale_to_world_shape=False, only_prediction=True, batch_size=inference_batch_size
                )
            else:
                # 3 dir for Peaks -> bad results
                seg_xyz, _ = direction_merger.get_seg_single_img_3_directions(
                    model, data=data, scale_to_world_shape=False, only_prediction=True, batch_size=inference_batch_size
                )
                seg = direction_merger.mean_fusion_peaks(seg_xyz, nr_cpus=nr_cpus)

            if peak_regression_part == "All":
                seg_all[:, :, :, (idx * len(config.CLASSES)) : (idx * len(config.CLASSES) + len(config.CLASSES))] = seg

        if peak_regression_part == "All":
            config.CLASSES = "All"
            len(config.CLASSES) = 3 * len(dataset_specific_utils.get_classes(config.CLASSES)[1:])
            seg = seg_all

    if config.TYPE_EXP == "tract_segmentation" and bundle_specific_postprocessing and not dropout_sampling:
        # Runtime ~4s
        seg = img_utils.bundle_specific_postprocessing(seg, dataset_specific_utils.get_classes(config.CLASSES)[1:])

    # runtime on HCP data: 5.1s
    seg = data_utils.cut_and_scale_img_back_to_original_img(seg, transformation, nr_cpus=nr_cpus)
    # runtime on HCP data: 1.6s
    seg = data_utils.add_original_zero_padding_again(seg, bbox, original_shape, len(config.CLASSES))

    if config.TYPE_EXP == "peak_regression":
        seg = peak_utils.mask_and_normalize_peaks(
            seg, tract_segmentations_path, dataset_specific_utils.get_classes(config.CLASSES)[1:], TOM_dilation, nr_cpus=nr_cpus
        )

    if config.TYPE_EXP == "tract_segmentation" and postprocess and not dropout_sampling:
        # Runtime ~7s for 1.25mm resolution
        # Runtime ~1.5s for  2mm resolution
        st = time.time()
        seg = img_utils.postprocess_segmentations(
            seg, dataset_specific_utils.get_classes(config.CLASSES)[1:], blob_thr=blob_size_thr, hole_closing=None
        )

    exp_utils.print_verbose(config.VERBOSE, "Took {}s".format(round(time.time() - start_time, 2)))
    return seg
