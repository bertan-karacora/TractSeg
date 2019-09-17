
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
import time
import pickle
import socket
import datetime
import numpy as np
from tqdm import tqdm
from pprint import pprint

from tractseg.libs import exp_utils
from tractseg.libs import metric_utils
from tractseg.libs import dataset_utils
from tractseg.libs import plot_utils
from tractseg.data.data_loader_inference import DataLoaderInference

def train_model(Config, model, data_loader):

    if Config.USE_VISLOGGER:
        try:
            from trixi.logger.visdom import PytorchVisdomLogger
        except ImportError:
            pass
        trixi = PytorchVisdomLogger(port=8080, auto_start=True)

    exp_utils.print_and_save(Config, socket.gethostname())

    epoch_times = []
    nr_of_updates = 0

    metrics = {}
    for type in ["train", "test", "validate"]:
        # metrics_new = {}
        for metric in Config.METRIC_TYPES:
            #todo: metrics.update(....) ?
            # metrics_new[metric + "_" + type] = [0]
            #todo: This should work
            metrics[metric + "_" + type] = [0]
        #todo: document
        # metrics = dict(list(metrics.items()) + list(metrics_new.items()))

    batch_gen_train = data_loader.get_batch_generator(batch_size=Config.BATCH_SIZE, type="train",
                                                      subjects=getattr(Config, "TRAIN_SUBJECTS"))
    batch_gen_val = data_loader.get_batch_generator(batch_size=Config.BATCH_SIZE, type="validate",
                                                    subjects=getattr(Config, "VALIDATE_SUBJECTS"))

    for epoch_nr in range(Config.NUM_EPOCHS):
        start_time = time.time()
        # current_lr = Config.LEARNING_RATE * (Config.LR_DECAY ** epoch_nr)
        # current_lr = Config.LEARNING_RATE

        # todo: use time default dict?
        data_preparation_time = 0
        network_time = 0
        metrics_time = 0
        saving_time = 0
        plotting_time = 0

        # todo: use defaultdict ?
        batch_nr = {
            "train": 0,
            "test": 0,
            "validate": 0
        }

        # todo: move to own function
        if Config.LOSS_WEIGHT is None:
            weight_factor = None
        elif Config.LOSS_WEIGHT_LEN == -1:
            weight_factor = float(Config.LOSS_WEIGHT)
        else:
            # Linearly decrease from LOSS_WEIGHT to 1 over LOSS_WEIGHT_LEN epochs
            if epoch_nr < Config.LOSS_WEIGHT_LEN:
                weight_factor = -((Config.LOSS_WEIGHT-1) /
                                  float(Config.LOSS_WEIGHT_LEN)) * epoch_nr + float(Config.LOSS_WEIGHT)
            else:
                weight_factor = 1.
            exp_utils.print_and_save(Config, "Current weight_factor: {}".format(weight_factor))

        if Config.ONLY_VAL:
            types = ["validate"]
        else:
            types = ["train", "validate"]

        for type in types:
            print_loss = []

            if Config.DIM == "2D":
                nr_of_samples = len(getattr(Config, type.upper() + "_SUBJECTS")) * Config.INPUT_DIM[0]
            else:
                nr_of_samples = len(getattr(Config, type.upper() + "_SUBJECTS"))

            # *Config.EPOCH_MULTIPLIER needed to have roughly same number of updates/batches as with 2D U-Net
            nr_batches = int(int(nr_of_samples / Config.BATCH_SIZE) * Config.EPOCH_MULTIPLIER)

            print("Start looping batches...")
            start_time_batch_part = time.time()
            for i in range(nr_batches):

                if type == "train":
                    batch = next(batch_gen_train)
                else:
                    batch = next(batch_gen_val)

                start_time_data_preparation = time.time()
                batch_nr[type] += 1

                x = batch["data"]  # (bs, nr_of_channels, x, y)
                y = batch["seg"]  # (bs, nr_of_classes, x, y)

                data_preparation_time += time.time() - start_time_data_preparation
                start_time_network = time.time()
                if type == "train":
                    nr_of_updates += 1
                    probs, metr_batch = model.train(x, y, weight_factor=weight_factor)
                elif type == "validate":
                    probs, metr_batch = model.test(x, y, weight_factor=weight_factor)
                elif type == "test":
                    probs, metr_batch = model.test(x, y, weight_factor=weight_factor)
                network_time += time.time() - start_time_network

                start_time_metrics = time.time()

                # move to extra function?
                if Config.CALC_F1:
                    if Config.EXPERIMENT_TYPE == "peak_regression":
                        peak_f1_mean = np.array([s.to('cpu') for s in list(metr_batch["f1_macro"].values())]).mean()
                        metr_batch["f1_macro"] = peak_f1_mean

                        metrics = metric_utils.add_to_metrics(metrics, metr_batch, type, Config.METRIC_TYPES)

                    else:
                        metr_batch["f1_macro"] = np.mean(metr_batch["f1_macro"])
                        metrics = metric_utils.add_to_metrics(metrics, metr_batch, type, Config.METRIC_TYPES)

                else:
                    metrics = metric_utils.calculate_metrics_onlyLoss(metrics, metr_batch["loss"], type=type)

                metrics_time += time.time() - start_time_metrics

                print_loss.append(metr_batch["loss"])
                if batch_nr[type] % Config.PRINT_FREQ == 0:
                    time_batch_part = time.time() - start_time_batch_part
                    start_time_batch_part = time.time()
                    exp_utils.print_and_save(Config,
                                             "{} Ep {}, Sp {}, loss {}, t print {}s, "
                                             "t batch {}s".format(type, epoch_nr,
                                                                  batch_nr[type] * Config.BATCH_SIZE,
                                                                  round(np.array(print_loss).mean(), 6),
                                                                  round(time_batch_part, 3),
                                                                  round( time_batch_part / Config.PRINT_FREQ, 3)))
                    print_loss = []

                if Config.USE_VISLOGGER:
                    plot_utils.plot_result_trixi(trixi, x, y, probs, metr_batch["loss"], metr_batch["f1_macro"], epoch_nr)


        ################################### Post Training tasks (each epoch) ###################################

        if Config.ONLY_VAL:
            metrics = metric_utils.normalize_last_element(metrics, batch_nr["validate"], type="validate")
            print("f1 macro validate: {}".format(round(metrics["f1_macro_validate"][0], 4)))
            return model

        # Average loss per batch over entire epoch
        metrics = metric_utils.normalize_last_element(metrics, batch_nr["train"], type="train")
        metrics = metric_utils.normalize_last_element(metrics, batch_nr["validate"], type="validate")
        # metrics = metric_utils.normalize_last_element(metrics, batch_nr["test"], type="test")

        print("  Epoch {}, Average Epoch loss = {}".format(epoch_nr, metrics["loss_train"][-1]))
        print("  Epoch {}, nr_of_updates {}".format(epoch_nr, nr_of_updates))

        # Adapt LR
        if Config.LR_SCHEDULE:
            if Config.LR_SCHEDULE_MODE == "min":
                model.scheduler.step(metrics["loss_validate"][-1])
            else:
                model.scheduler.step(metrics["f1_macro_validate"][-1])
            model.print_current_lr()

        # Save Weights
        start_time_saving = time.time()
        if Config.SAVE_WEIGHTS:
            model.save_model(metrics, epoch_nr, mode=Config.BEST_EPOCH_SELECTION)
        saving_time += time.time() - start_time_saving

        # Create Plots
        start_time_plotting = time.time()
        pickle.dump(metrics, open(join(Config.EXP_PATH, "metrics.pkl"), "wb"))
        plot_utils.create_exp_plot(metrics, Config.EXP_PATH, Config.EXP_NAME,
                                   keys=["loss", "f1_macro"],
                                   types=["train", "validate"],
                                   selected_ax=["loss", "f1"],
                                   fig_name="metrics_all.png")
        plot_utils.create_exp_plot(metrics, Config.EXP_PATH, Config.EXP_NAME, without_first_epochs=True,
                                   keys=["loss", "f1_macro"],
                                   types=["train", "validate"],
                                   selected_ax=["loss", "f1"],
                                   fig_name="metrics.png")
        if "angle_err" in Config.METRIC_TYPES:
            plot_utils.create_exp_plot(metrics, Config.EXP_PATH, Config.EXP_NAME, without_first_epochs=True,
                                       keys=["loss", "angle_err"],
                                       types=["train", "validate"],
                                       selected_ax=["loss", "f1"],
                                       fig_name="metrics_angle.png")

        plotting_time += time.time() - start_time_plotting

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        exp_utils.print_and_save(Config, "  Epoch {}, time total {}s".format(epoch_nr, epoch_time))
        exp_utils.print_and_save(Config, "  Epoch {}, time UNet: {}s".format(epoch_nr, network_time))
        exp_utils.print_and_save(Config, "  Epoch {}, time metrics: {}s".format(epoch_nr, metrics_time))
        exp_utils.print_and_save(Config, "  Epoch {}, time saving files: {}s".format(epoch_nr, saving_time))
        exp_utils.print_and_save(Config, str(datetime.datetime.now()))

        # Adding next Epoch
        if epoch_nr < Config.NUM_EPOCHS-1:
            metrics = metric_utils.add_empty_element(metrics)


    ####################################
    # After all epochs
    ###################################
    with open(join(Config.EXP_PATH, "Hyperparameters.txt"), "a") as f:  # a for append
        f.write("\n\n")
        f.write("Average Epoch time: {}s".format(sum(epoch_times) / float(len(epoch_times))))


def predict_img(Config, model, data_loader, probs=False, scale_to_world_shape=True, only_prediction=False,
                batch_size=1):
    """
    Runtime on CPU
    - python 2 + pytorch 0.4:
      bs=1  -> 9min      around 4.5GB RAM (maybe even 7GB)
      bs=48 -> 6.5min           30GB RAM
    - python 3 + pytorch 1.0:
      bs=1  -> 2.7min    around 7GB RAM

    Args:
        Config:
        model:
        data_loader:
        probs:
        scale_to_world_shape:
        only_prediction:
        batch_size:

    Returns:

    """

    #todo add _ to helper functions
    def _finalize_data(layers):
        layers = np.array(layers)

        if Config.DIM == "2D":
            # Get in right order (x,y,z) and
            if Config.SLICE_DIRECTION == "x":
                layers = layers.transpose(0, 1, 2, 3)

            elif Config.SLICE_DIRECTION == "y":
                layers = layers.transpose(1, 0, 2, 3)

            elif Config.SLICE_DIRECTION == "z":
                layers = layers.transpose(1, 2, 0, 3)

        if scale_to_world_shape:
            layers = dataset_utils.scale_input_to_original_shape(layers, Config.DATASET, Config.RESOLUTION)

        #todo: move to top of function
        assert (layers.dtype == np.float32)  # .astype() quite slow -> use assert to make sure type is right
        return layers

    img_shape = [Config.INPUT_DIM[0], Config.INPUT_DIM[0], Config.INPUT_DIM[0], Config.NR_OF_CLASSES]
    layers_seg = np.empty(img_shape).astype(np.float32)
    layers_y = None if only_prediction else np.empty(img_shape).astype(np.float32)
    batch_generator = data_loader.get_batch_generator(batch_size=batch_size)
    batch_generator = list(batch_generator)
    idx = 0
    for batch in tqdm(batch_generator):
        x = batch["data"]   # (bs, nr_of_channels, x, y)
        y = batch["seg"]    # (bs, nr_of_classes, x, y)
        y = y.numpy()

        if not only_prediction:
            y = y.astype(Config.LABELS_TYPE)
            # y = np.squeeze(y)   # remove bs dimension which is only 1 -> (nrClasses, x, y)
            if Config.DIM == "2D":
                y = y.transpose(0, 2, 3, 1) # (bs, x, y, nr_of_classes)
            else:
                y = y.transpose(0, 2, 3, 4, 1)

        if Config.DROPOUT_SAMPLING:
            #For Dropout Sampling (must set Deterministic=False in model)
            NR_SAMPLING = 30
            samples = []
            for i in range(NR_SAMPLING):
                layer_probs = model.predict(x)  # (bs, x, y, nrClasses)
                samples.append(layer_probs)

            samples = np.array(samples)  # (NR_SAMPLING, bs, x, y, nrClasses)
            layer_probs = np.std(samples, axis=0)    # (bs,x,y,nrClasses)
        else:
            # For normal prediction
            layer_probs = model.predict(x)  # (bs, x, y, nrClasses)

        if probs:
            seg = layer_probs   # (x, y, nrClasses)
        else:
            seg = layer_probs
            seg[seg >= Config.THRESHOLD] = 1
            seg[seg < Config.THRESHOLD] = 0
            seg = seg.astype(np.uint8)

        if Config.DIM == "2D":
            layers_seg[idx*batch_size:(idx+1)*batch_size, :, :, :] = seg
            if not only_prediction:
                layers_y[idx*batch_size:(idx+1)*batch_size, :, :, :] = y
        else:
            layers_seg = np.squeeze(seg)
            if not only_prediction:
                layers_y = np.squeeze(y)

        idx += 1

    layers_seg = _finalize_data(layers_seg)
    if not only_prediction:
        layers_y = _finalize_data(layers_y)
    return layers_seg, layers_y   # (Prediction, Groundtruth)


def test_whole_subject(Config, model, subjects, type):

    metrics = {
        "loss_" + type: [0],
        "f1_macro_" + type: [0],
    }

    # Metrics per bundle
    metrics_bundles = {}
    for bundle in exp_utils.get_bundle_names(Config.CLASSES)[1:]:
        metrics_bundles[bundle] = [0]

    for subject in subjects:
        print("{} subject {}".format(type, subject))
        start_time = time.time()

        data_loader = DataLoaderInference(Config, subject=subject)
        img_probs, img_y = predict_img(Config, model, data_loader, probs=True)
        # img_probs_xyz, img_y = DirectionMerger.get_seg_single_img_3_directions(Config, model, subject=subject)
        # img_probs = DirectionMerger.mean_fusion(Config.THRESHOLD, img_probs_xyz, probs=True)

        print("Took {}s".format(round(time.time() - start_time, 2)))

        if Config.EXPERIMENT_TYPE == "peak_regression":
            f1 = metric_utils.calc_peak_length_dice(Config, img_probs, img_y,
                                                    max_angle_error=Config.PEAK_DICE_THR,
                                                    max_length_error=Config.PEAK_DICE_LEN_THR)
            peak_f1_mean = np.array([s for s in f1.values()]).mean()  # if f1 for multiple bundles
            metrics = metric_utils.calculate_metrics(metrics, None, None, 0, f1=peak_f1_mean,
                                                     type=type, threshold=Config.THRESHOLD)
            metrics_bundles = metric_utils.calculate_metrics_each_bundle(metrics_bundles, None, None,
                                                                         exp_utils.get_bundle_names(Config.CLASSES)[1:],
                                                                         f1, threshold=Config.THRESHOLD)
        else:
            img_probs = np.reshape(img_probs, (-1, img_probs.shape[-1]))  #Flatten all dims except nrClasses dim
            img_y = np.reshape(img_y, (-1, img_y.shape[-1]))
            metrics = metric_utils.calculate_metrics(metrics, img_y, img_probs, 0,
                                                     type=type, threshold=Config.THRESHOLD)
            metrics_bundles = metric_utils.calculate_metrics_each_bundle(metrics_bundles, img_y, img_probs,
                                                                         exp_utils.get_bundle_names(Config.CLASSES)[1:],
                                                                         threshold=Config.THRESHOLD)

    metrics = metric_utils.normalize_last_element(metrics, len(subjects), type=type)
    metrics_bundles = metric_utils.normalize_last_element_general(metrics_bundles, len(subjects))

    print("WHOLE SUBJECT:")
    pprint(metrics)
    print("WHOLE SUBJECT BUNDLES:")
    pprint(metrics_bundles)

    with open(join(Config.EXP_PATH, "score_" + type + "-set.txt"), "w") as f:
        pprint(metrics, f)
        f.write("\n\nWeights: {}\n".format(Config.WEIGHTS_PATH))
        f.write("type: {}\n\n".format(type))
        pprint(metrics_bundles, f)
    pickle.dump(metrics, open(join(Config.EXP_PATH, "score_" + type + ".pkl"), "wb"))
    return metrics