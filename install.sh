#!/bin/bash

# Setup git user, ssh keys, clone repo, apt update, apt upgrade etc.

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
