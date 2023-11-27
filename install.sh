#!/bin/bash

# For local development:
# Fork repository.
# Install git, configure git and ssh keys / access token / ...
# Clone repo.

sudo add-apt-repository ppa:deadsnakes/
sudo apt update
cat requirements_apt.txt | xargs sudo apt install -y

python3.8 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install wheel
pip install -r requirements_pip.txt
pip install --no-cache-dir -e .

git clone https://github.com/MRtrix3/mrtrix3.git
cd mrtrix3
./configure
# You might need:
# NUMBER_OF_PROCESSORS=1
./build
./set_path
# Check via:
# dwi2response -help

# You probably also need to install FSL (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSL).
# It is free, you need to register at their website.
# Run:
# python fslinstaller.py
# You might need:
# export FSLDIR=~/TractSeg/fsl
# Check via:
# bet -help
