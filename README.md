# TractSeg

TractSeg is a tool for fast and accurate white matter bundle segmentation from Diffusion MRI. The original implementation segments in fields of fODF peaks. This repository contains the code for running experiments using a fourth-order tensor representation of the fODF or approximations thereof.
For a documentation of TractSeg please refer to the [original code repository](https://github.com/MIC-DKFZ/TractSeg) as well as the following publications by Wasserthal et al.:

* [TractSeg - Fast and accurate white matter tract segmentation](https://doi.org/10.1016/j.neuroimage.2018.07.070) ([free arxiv version](https://arxiv.org/abs/1805.07103))
[NeuroImage 2018]
* [Combined tract segmentation and orientation mapping for bundle-specific tractography](https://www.sciencedirect.com/science/article/pii/S136184151930101X)
[Medical Image Analysis 2019]

## Table of contents

- [TractSeg](#tractseg)
  - [Table of contents](#table-of-contents)
  - [Installation](#installation)
  - [Train your own model](#train-your-own-model)

## Installation

This code is known to run on Linux and Python >= 3.9.
To install the original distributed version of TractSeg, use

```
pip install TractSeg
```

To install this framework for running experiments with alternative inputs, clone this repository and run the installation script:

```
chmod +x ./install
./install
```

Finally, please adapt the contents of `tractseg/config.yaml` to setup system paths.

## Train your own model

1. Download and prepare the HCP data. Your data directory should contain at least the following files:

    ```
    custom_path/subject_id/
          '-> T1w_acpc_dc_restore_1.25.nii.gz
          '-> Diffusion/
                '-> bvals
                '-> bvecs
                '-> data.nii.gz
                '-> nodif_brain_mask.nii.gz
    ```  

2. Run the setup and data processing script. This will take a while and prepare all experiments.

    ```
    chmod +x ./setup_experiments
    ./setup_experiments
    ```

3. Create a custom experiment config or use one of the prepared ones under `tractseg/config/custom/`.

4. Run `ExpRunner --path_config_exp path/to/your/experiment/config --train --validate --test --segmentations`. You may want to have a look at additional arguments of this script.

5. Under the path you specified in `tractseg/config.yaml` you will find the results.
