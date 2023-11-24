#!/bin/bash

# For local development:
# Fork repository
# Install git, configure git and ssh keys / access token / ...
# Clone repo

cat requirements_apt.txt | xargs sudo apt install -y
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install wheel
pip install -r requirements_pip.txt
pip install --no-cache-dir -e .

git clone https://github.com/MRtrix3/mrtrix3.git
cd mrtrix3
./configure
./build
./set_path

# You probably also need to install FSL (https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSL).
# For this, register yourself at their website.
# python fslinstaller.py
