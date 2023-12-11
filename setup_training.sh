#!/bin/bash

download_data() {
    echo "Donwloading data..."

    wget https://zenodo.org/records/1477956/files/HCP105_Zenodo_NewTrkFormat.zip

    echo "Donwloading data finished."
}

install_batchgenerators() {
    echo "Installing batchgenerators..."

    git clone https://github.com/MIC-DKFZ/batchgenerators.git
    pip install -e batchgenerators

    echo "Installing batchgenerators finished."
}

binarize_tracts() {
    echo "Binarizing tracts..."

    dir_data="data/HCP"
    dir_ignore="5ttgen-tmp-PMAWPS"

    for dir_subject in "$dir_data"/*; do
        if [ "$dir_subject" != "$dir_data/$dir_ignore" ]; then
            id_subject=$(basename "$dir_subject")
            echo "Binarizing tracts for subject $id_subject"

            dir_out="$dir_subject/tracts_binarized"
            mkdir -p "$dir_out"

            for file_trk in "$dir_subject/tracts"/*.trk; do
                name_tract=$(basename "${file_trk%.*}")

                echo "Binarizing $name_tract for subject $id_subject"

                python tractseg/utils/trk_2_binary.py \
                    -i "$file_trk" \
                    -o "$dir_out/$name_tract.nii.gz" \
                    --ref "$dir_subject/Diffusion/nodif_brain_mask.nii.gz" &
            done
            wait
        fi
    done

    echo "Binarizing tracts finished."
}

main() {
    # download_data
    # install_batchgenerators
    binarize_tracts
}

# May need to run with
# sudo -E env PATH=$PATH ./setup_training.sh
main

# data2 script for fODFs
# ExpRunner --config my_custom_experiment
