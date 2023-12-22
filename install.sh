#!/bin/bash

num_processors_building_mrtrix3=1
dir_fsl=~/TractSeg/libs/fsl

install_system_requirements() {
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install -y python3.9 python3.9-venv python3.9-dev python-is-python3
    sudo apt install -y $(cat requirements/requirements_apt.txt)
}

activate_python_venv() {
    python3.9 -m venv .venv
    source .venv/bin/activate
}

install_python_requirements() {
    pip install --upgrade pip
    pip install wheel
    pip install -r requirements/requirements_pip.txt
    pip install -e .
}

install_mrtrix3() {
    export NUMBER_OF_PROCESSORS=$num_processors_building_mrtrix3
    cd libs
    git clone https://github.com/MRtrix3/mrtrix3.git
    cd mrtrix3
    ./configure
    ./build
    ./set_path
    cd ../..
}

install_fsl() {
    export FSLDIR=$dir_fsl
    cd libs
    wget https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fslinstaller.py
    python fslinstaller.py
    rm fslinstaller.py
}

install_explicit_dependencies() {
    mkdir -p libs
    install_mrtrix3
    install_fsl
}

main() {
    install_system_requirements
    activate_python_venv
    install_python_requirements
    install_explicit_dependencies
}

main
