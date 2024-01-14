import importlib_resources
from pathlib import Path

import pprint
import yaml

PATH_CONFIG_SYSTEM = str(importlib_resources.files("tractseg") / "config.yaml")
PATH_CONFIG_EXP = None


def get_path_config_pretrained(input_type, output_type, dropout_sampling=False, tract_definition="TractQuerier+"):
    mapping_config = {
        ("TractQuerier+", "peaks", "tract_segmentation", False): "TractSeg_PeakRot4.yaml",
        ("TractQuerier+", "peaks", "tract_segmentation", True): "TractSeg_PeakRot4.yaml",
        ("TractQuerier+", "peaks", "endings_segmentation", False): "EndingsSeg_PeakRot4.yaml",
        ("TractQuerier+", "peaks", "TOM", False): "Peaks_AngL.yaml",
        ("TractQuerier+", "peaks", "dm_regression", False): "DmReg.yaml",
        ("TractQuerier+", "T1", "tract_segmentation", False): "TractSeg_T1_125mm_DAugAll.yaml",
        ("TractQuerier+", "T1", "endings_segmentation", False): "EndingsSeg_12g90g270g_125mm_DAugAll.yaml",
        ("xtract", "peaks", "tract_segmentation", False): "TractSeg_All_xtract_PeakRot4.yaml",
        ("xtract", "peaks", "tract_segmentation", True): "TractSeg_All_xtract_PeakRot4.yaml",
        ("xtract", "peaks", "dm_regression", False): "DmReg_All_xtract_PeakRot4.yaml",
    }

    if (tract_definition, input_type, output_type, dropout_sampling) in mapping_config:
        filename_config_experiment = mapping_config[(tract_definition, input_type, output_type, dropout_sampling)]
    else:
        raise ValueError(f"ERROR: Unsupported combination: {tract_definition}, {input_type}, {output_type}, {dropout_sampling}")

    path = importlib_resources.files("tractseg.experiments.pretrained_models") / filename_config_experiment
    return path


def read_config(path):
    with open(path, "r") as stream:
        try:
            config = yaml.safe_load(stream) or {}
        except yaml.YAMLError as e:
            print(e)

    return config


def set_attributes(config):
    # Use module globals for the attributes.
    # Globals are generally to be avoided but actually considered best practice for a global package config.
    # See: https://stackoverflow.com/questions/5055042/whats-the-best-practice-using-a-settings-file-in-python
    # See: https://stackoverflow.com/questions/30556857/creating-a-static-class-with-no-instances
    # Note: This is not secure whatsoever (but the original code isn't either).
    for key, value in config.items():
        globals()[key.upper()] = value


def set_config_system():
    config_system = read_config(Path(PATH_CONFIG_SYSTEM))
    set_attributes(config_system)


def set_config_exp(path):
    globals()["PATH_CONFIG_EXP"] = str(path)
    config_exp = read_config(Path(PATH_CONFIG_EXP))
    set_attributes(config_exp)


def dump():
    d = {k: v for k, v in globals().items() if k.isupper()}
    pprint.pprint(d)


def save(path):
    d = {k: v for k, v in globals().items() if k.isupper()}
    with open(path / "config_exp.yaml", "w+") as file:
        yaml.dump(d, file, default_flow_style=False)


set_config_system()
