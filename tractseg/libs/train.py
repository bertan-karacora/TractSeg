from os.path import join
import time
import pickle
import datetime
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from pprint import pprint

import tractseg.config as config
from tractseg.libs import direction_merger
from tractseg.libs import exp_utils
from tractseg.libs import metric_utils
from tractseg.libs import plot_utils
from tractseg.data.data_loader_inference import DataLoaderInference
from tractseg.data.data_loader_training import DataLoaderTraining


def _get_weights_for_this_epoch(epoch_nr):
    if config.LOSS_WEIGHT is None:
        weight_factor = None
    elif config.LOSS_WEIGHT_LEN == -1:
        weight_factor = float(config.LOSS_WEIGHT)
    else:
        # Linearly decrease from LOSS_WEIGHT to 1 over LOSS_WEIGHT_LEN epochs
        if epoch_nr < config.LOSS_WEIGHT_LEN:
            weight_factor = -((config.LOSS_WEIGHT - 1) / float(config.LOSS_WEIGHT_LEN)) * epoch_nr + float(config.LOSS_WEIGHT)
        else:
            weight_factor = 1.0
        exp_utils.print_and_save(config.PATH_EXP, "Current weight_factor: {}".format(weight_factor))
    return weight_factor


def _update_metrics(calc_f1, TYPE_EXP, metric_types, metrics, metr_batch, type):
    if calc_f1:
        if TYPE_EXP == "peak_regression":
            peak_f1_mean = np.array([s.to("cpu") for s in list(metr_batch["f1_macro"].values())]).mean()
            metr_batch["f1_macro"] = peak_f1_mean

            metrics = metric_utils.add_to_metrics(metrics, metr_batch, type, metric_types)

        else:
            metr_batch["f1_macro"] = np.mean(metr_batch["f1_macro"])
            metrics = metric_utils.add_to_metrics(metrics, metr_batch, type, metric_types)

    else:
        metrics = metric_utils.calculate_metrics_onlyLoss(metrics, metr_batch["loss"], type=type)
    return metrics


def train_model(model):
    if config.USE_VISLOGGER:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(config.PATH_EXP)

    data_loader = DataLoaderTraining()
    batch_gen_train = data_loader.get_batch_generator(batch_size=config.BATCH_SIZE, type="train", subjects=config.SUBJECTS_TRAIN)
    batch_gen_val = data_loader.get_batch_generator(batch_size=config.BATCH_SIZE, type="validate", subjects=config.SUBJECTS_VALIDATE)

    epoch_times = []
    nr_of_updates = 0

    metrics = {}
    for type in ["train", "test", "validate"]:
        for metric in config.METRIC_TYPES:
            metrics[metric + "_" + type] = [0]

    for epoch_nr in range(config.NUM_EPOCHS):
        start_time = time.time()

        timings = defaultdict(lambda: 0)
        batch_nr = defaultdict(lambda: 0)
        weight_factor = _get_weights_for_this_epoch(epoch_nr)
        types = []
        if config.TRAIN:
            types += ["train"]
        if config.VALIDATE:
            types += ["validate"]

        for type in types:
            print_loss = []

            if len(config.SHAPE_INPUT) == 2:
                nr_of_samples = len(getattr(config, f"SUBJECTS_{type.upper()}")) * config.SHAPE_INPUT[0]
            else:
                nr_of_samples = len(getattr(config, f"SUBJECTS_{type.upper()}"))

            # *config.EPOCH_MULTIPLIER needed to have roughly same number of updates/batches as with 2D U-Net
            nr_batches = int(int(nr_of_samples / config.BATCH_SIZE) * config.EPOCH_MULTIPLIER)

            print("Start looping batches...")
            start_time_batch_part = time.time()
            for i in range(nr_batches):
                batch = next(batch_gen_train) if type == "train" else next(batch_gen_val)

                start_time_data_preparation = time.time()
                batch_nr[type] += 1

                x = batch["data"]  # (batch_size, nr_of_channels, x, y)
                y = batch["seg"]  # (batch_size, len(config.CLASSES), x, y)

                timings["data_preparation_time"] += time.time() - start_time_data_preparation
                start_time_network = time.time()
                if type == "train":
                    nr_of_updates += 1
                    probs, metr_batch = model.train(x, y, weight_factor=weight_factor)
                elif type == "validate":
                    probs, metr_batch = model.test(x, y, weight_factor=weight_factor)
                timings["network_time"] += time.time() - start_time_network

                start_time_metrics = time.time()
                metrics = _update_metrics(config.CALC_F1, config.TYPE_EXP, config.METRIC_TYPES, metrics, metr_batch, type)
                timings["metrics_time"] += time.time() - start_time_metrics

                print_loss.append(metr_batch["loss"])
                if batch_nr[type] % config.PRINT_FREQ == 0:
                    time_batch_part = time.time() - start_time_batch_part
                    start_time_batch_part = time.time()
                    exp_utils.print_and_save(
                        config.PATH_EXP,
                        "{} Ep {}, Sp {}, loss {}, t print {}s, t batch {}s".format(
                            type,
                            epoch_nr,
                            batch_nr[type] * config.BATCH_SIZE,
                            round(np.array(print_loss).mean(), 6),
                            round(time_batch_part, 3),
                            round(time_batch_part / config.PRINT_FREQ, 3),
                        ),
                    )
                    print_loss = []

                if config.USE_VISLOGGER:
                    writer.add_scalar(f"Loss/{type}", metr_batch["loss"], epoch_nr * nr_batches + i)
                    writer.add_scalar(f"F1/{type}", np.mean(metr_batch["f1_macro"]), epoch_nr * nr_batches + i)

                    if i == nr_batches - 1:
                        num_images = 5
                        x_norm = (x - x.min()) / (x.max() - x.min() + 1e-7)
                        writer.add_images(f"Inputs/{type}", x_norm[:num_images, :3, :, :], epoch_nr)
                        writer.add_images(f"Predictions/{type}", probs[:num_images, 15:16, :, :], epoch_nr)

                        # Bundle: CST, GREEN: GT; RED: prediction (FP); YELLOW: prediction (TP)
                        combined = np.zeros((num_images, 3, y.shape[2], y.shape[3]))
                        combined[:, 0, :, :] = np.where(probs[:num_images, 15, :, :] > 0.5, 1, 0)  # Red
                        combined[:, 1, :, :] = y[:num_images, 15, :, :]  # Green
                        writer.add_images(f"Combined/{type}", combined, epoch_nr)
                    writer.flush()

        ################################### Post Training tasks (each epoch) ###################################

        # Average loss per batch over entire epoch
        metrics = metric_utils.normalize_last_element(metrics, batch_nr["train"], type="train")
        metrics = metric_utils.normalize_last_element(metrics, batch_nr["validate"], type="validate")

        print("  Epoch {}, Average Epoch loss = {}".format(epoch_nr, metrics["loss_train"][-1]))
        exp_utils.print_and_save(config.PATH_EXP, "  Epoch {}, nr_of_updates {}".format(epoch_nr, nr_of_updates))

        # Adapt LR
        if config.LR_SCHEDULE:
            if config.LR_SCHEDULE_MODE == "min":
                model.scheduler.step(metrics["loss_validate"][-1])
            else:
                model.scheduler.step(metrics["f1_macro_validate"][-1])
            model.print_current_lr()

        # Save Weights
        start_time_saving = time.time()
        if config.SAVE_WEIGHTS:
            model.save_model(metrics, epoch_nr, mode=config.BEST_EPOCH_SELECTION)
        timings["saving_time"] += time.time() - start_time_saving

        # Create Plots
        start_time_plotting = time.time()
        pickle.dump(metrics, open(join(config.PATH_EXP, "metrics.pkl"), "wb"))
        plot_utils.create_exp_plot(
            metrics,
            config.PATH_EXP,
            config.NAME_EXP,
            keys=["loss", "f1_macro"],
            types=["train", "validate"],
            selected_ax=["loss", "f1"],
            fig_name="metrics_all.png",
        )
        plot_utils.create_exp_plot(
            metrics,
            config.PATH_EXP,
            config.NAME_EXP,
            without_first_epochs=True,
            keys=["loss", "f1_macro"],
            types=["train", "validate"],
            selected_ax=["loss", "f1"],
            fig_name="metrics.png",
        )
        if "angle_err" in config.METRIC_TYPES:
            plot_utils.create_exp_plot(
                metrics,
                config.PATH_EXP,
                config.NAME_EXP,
                without_first_epochs=True,
                keys=["loss", "angle_err"],
                types=["train", "validate"],
                selected_ax=["loss", "f1"],
                fig_name="metrics_angle.png",
            )

        timings["plotting_time"] += time.time() - start_time_plotting

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        exp_utils.print_and_save(config.PATH_EXP, "  Epoch {}, time total {}s".format(epoch_nr, epoch_time))
        exp_utils.print_and_save(config.PATH_EXP, "  Epoch {}, time UNet: {}s".format(epoch_nr, timings["network_time"]))
        exp_utils.print_and_save(config.PATH_EXP, "  Epoch {}, time metrics: {}s".format(epoch_nr, timings["metrics_time"]))
        exp_utils.print_and_save(config.PATH_EXP, "  Epoch {}, time saving files: {}s".format(epoch_nr, timings["saving_time"]))
        exp_utils.print_and_save(config.PATH_EXP, str(datetime.datetime.now()))

        # Adding next Epoch
        if epoch_nr < config.NUM_EPOCHS - 1:
            metrics = metric_utils.add_empty_element(metrics)

        if config.USE_VISLOGGER:
            writer.add_scalar("Time total", epoch_time, epoch_nr)
            writer.add_scalar("Time network", timings["network_time"], epoch_nr)

    if config.USE_VISLOGGER:
        writer.close()

    print("Average Epoch time: {}s".format(sum(epoch_times) / float(len(epoch_times))))


def predict_img(model, data_loader, probs=False, scale_to_world_shape=True, only_prediction=False, batch_size=1, unit_test=False):
    """
    Return predictions for one 3D image.

    Runtime on CPU
    - python 2 + pytorch 0.4:
          bs=1  -> 9min      ~7GB RAM
          bs=48 -> 6.5min    ~30GB RAM
    - python 3 + pytorch 1.0:
          bs=1  -> 2.7min    ~7GB RAM
    """

    def _finalize_data(layers):
        layers = np.array(layers)

        if len(config.SHAPE_INPUT) == 2:
            # Get in right order (x,y,z) and
            if config.SLICE_DIRECTION == "x":
                layers = layers.transpose(0, 1, 2, 3)

            elif config.SLICE_DIRECTION == "y":
                layers = layers.transpose(1, 0, 2, 3)

            elif config.SLICE_DIRECTION == "z":
                layers = layers.transpose(1, 2, 0, 3)

        # if scale_to_world_shape:
        #     layers = dataset_specific_utils.scale_input_to_original_shape(layers, config.DATASET, config.RESOLUTION)

        assert layers.dtype == np.float32
        return layers

    img_shape = [config.SHAPE_INPUT[0], config.SHAPE_INPUT[0], config.SHAPE_INPUT[0], len(config.CLASSES)]
    layers_seg = np.empty(img_shape).astype(np.float32)
    layers_y = None if only_prediction else np.empty(img_shape).astype(np.float32)

    if unit_test:
        # Return some mockup data to test different input arguments end 2 end and to test the postprocessing of the
        # segmentations (using real segmentations on DWI test image would take too much time if we run it for
        # different configurations)
        probs = np.zeros(img_shape).astype(np.float32)

        # CA (bundle specific postprocessing)
        probs[10:30, 10:30, 10:30, 4] = 0.7  # big blob 1
        probs[10:30, 10:30, 40:50, 4] = 0.7  # big blob 2
        probs[20:25, 20:25, 30:34, 4] = 0.4  # incomplete bridge between blobs with lower probability
        probs[20:25, 20:25, 36:40, 4] = 0.4  # incomplete bridge between blobs with lower probability
        probs[50:55, 50:55, 50:55, 4] = 0.2  # below threshold
        probs[60:63, 60:63, 60:63, 4] = 0.9  # small blob -> will get removed by postprocessing
        # should restore the bridge

        # CC_1
        probs[10:30, 10:30, 10:30, 5] = 0.7  # big blob 1
        probs[10:30, 10:30, 40:50, 5] = 0.7  # big blob 2
        probs[20:25, 20:25, 30:34, 5] = 0.4  # incomplete bridge between blobs with lower probability
        probs[20:25, 20:25, 36:40, 5] = 0.4  # incomplete bridge between blobs with lower probability
        probs[50:55, 50:55, 50:55, 5] = 0.2  # below threshold
        probs[60:63, 60:63, 60:63, 5] = 0.9  # small blob -> will get removed by postprocessing
        # should not restore the bridge

        return probs, layers_y

    batch_generator = data_loader.get_batch_generator(batch_size=batch_size)
    batch_generator = list(batch_generator)
    idx = 0
    for batch in tqdm(batch_generator):
        x = batch["data"]  # (bs, nr_channels, x, y)
        y = batch["seg"]  # (bs, nr_classes, x, y)
        y = y.numpy()

        if not only_prediction:
            y = y.astype(exp_utils.get_type_labels(config.TYPE_LABELS))
            if len(config.SHAPE_INPUT) == 2:
                y = y.transpose(0, 2, 3, 1)  # (bs, x, y, nr_classes)
            else:
                y = y.transpose(0, 2, 3, 4, 1)

        if config.DROPOUT_SAMPLING:
            # For Dropout Sampling (must set deterministic=False in model)
            NR_SAMPLING = 30
            samples = []
            for i in range(NR_SAMPLING):
                layer_probs = model.predict(x)  # (bs, x, y, nr_classes)
                samples.append(layer_probs)

            samples = np.array(samples)  # (NR_SAMPLING, bs, x, y, nr_classes)
            layer_probs = np.std(samples, axis=0)  # (bs, x, y, nr_classes)
        else:
            # For normal prediction
            layer_probs = model.predict(x)  # (bs, x, y, nr_classes)

        if probs:
            seg = layer_probs  # (x, y, nr_classes)
        else:
            seg = layer_probs
            seg[seg >= config.THRESHOLD] = 1
            seg[seg < config.THRESHOLD] = 0
            seg = seg.astype(np.uint8)

        if len(config.SHAPE_INPUT) == 2:
            layers_seg[idx * batch_size : (idx + 1) * batch_size, :, :, :] = seg
            if not only_prediction:
                layers_y[idx * batch_size : (idx + 1) * batch_size, :, :, :] = y
        else:
            layers_seg = np.squeeze(seg)
            if not only_prediction:
                layers_y = np.squeeze(y)

        idx += 1

    layers_seg = _finalize_data(layers_seg)
    if not only_prediction:
        layers_y = _finalize_data(layers_y)
    return layers_seg, layers_y


def test_whole_subject(model, subjects, type):
    metrics = {
        "loss_" + type: [0],
        "f1_macro_" + type: [0],
    }

    metrics_bundles = defaultdict(lambda: [0])

    for subject in subjects:
        print(f"{type} subject {subject}")
        start_time = time.time()

        data_loader = DataLoaderInference(subject=subject)
        # img_probs, img_y = predict_img(model, data_loader, probs=True)
        img_probs_xyz, img_y = direction_merger.get_seg_single_img_3_directions(model, subject=subject)
        img_probs = direction_merger.mean_fusion(config.THRESHOLD, img_probs_xyz, probs=True)

        print("Took {}s".format(round(time.time() - start_time, 2)))

        if config.TYPE_EXP == "peak_regression":
            f1 = metric_utils.calc_peak_length_dice(
                config.CLASSSET, img_probs, img_y, max_angle_error=config.PEAK_DICE_THR, max_length_error=config.PEAK_DICE_LEN_THR
            )
            peak_f1_mean = np.array([s for s in f1.values()]).mean()  # if f1 for multiple bundles
            metrics = metric_utils.calculate_metrics(metrics, None, None, 0, f1=peak_f1_mean, type=type, threshold=config.THRESHOLD)
            metrics_bundles = metric_utils.calculate_metrics_each_bundle(metrics_bundles, None, None, config.CLASSES, f1, threshold=config.THRESHOLD)
        else:
            img_probs = np.reshape(img_probs, (-1, img_probs.shape[-1]))  # Flatten all dims except nr_classes dim
            img_y = np.reshape(img_y, (-1, img_y.shape[-1]))
            metrics = metric_utils.calculate_metrics(metrics, img_y, img_probs, 0, type=type, threshold=config.THRESHOLD)
            metrics_bundles = metric_utils.calculate_metrics_each_bundle(
                metrics_bundles, img_y, img_probs, config.CLASSES, threshold=config.THRESHOLD
            )

    metrics = metric_utils.normalize_last_element(metrics, len(subjects), type=type)
    metrics_bundles = metric_utils.normalize_last_element_general(metrics_bundles, len(subjects))

    print("WHOLE SUBJECT:")
    pprint(metrics)
    print("WHOLE SUBJECT BUNDLES:")
    pprint(metrics_bundles)

    with open(join(config.PATH_EXP, "score_" + type + "-set.txt"), "w") as f:
        pprint(metrics, f)
        pprint(metrics_bundles, f)
    pickle.dump(metrics, open(join(config.PATH_EXP, "score_" + type + ".pkl"), "wb"))
    return metrics
