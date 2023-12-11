#!/bin/bash

download_data() {
    echo "Donwloading data..."

    wget https://zenodo.org/records/1477956/files/HCP105_Zenodo_NewTrkFormat.zip

    echo "Downloading data finished."
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

    ids_subject=(599469 599671 601127 613538 620434 622236 623844 627549 638049 644044 645551 654754 665254 672756 673455 677968 679568 680957 683256 685058 687163 690152 695768 702133 704238 705341 709551 715041 715647 729254 729557 732243 734045 742549 748258 748662 749361 751348 753251 756055 759869 761957 765056 770352 771354 779370 782561 784565 786569 789373 792564 792766 802844 814649 816653 826353 826454 833148 833249 837560 837964 845458 849971 856766 857263 859671 861456 865363 871762 871964 872158 872764 877168 877269 887373 889579 894673 896778 896879 898176 899885 901038 901139 901442 904044 907656 910241 912447 917255 922854 930449 932554 951457 957974 958976 959574 965367 965771 978578 979984 983773 984472 987983 991267 992774)

    for id_subject in "${ids_subject[@]}"; do
        echo "Binarizing tracts for subject $id_subject"

        dir_out="$dir_data/$id_subject/tracts_binarized"
        mkdir -p "$dir_out"

        for file_trk in "$dir_data/$id_subject/tracts"/*.trk; do
            name_tract=$(basename "${file_trk%.*}")

            echo "Binarizing $name_tract for subject $id_subject"

            python tractseg/utils/trk_2_binary.py \
                -i "$file_trk" \
                -o "$dir_out/$name_tract.nii.gz" \
                --ref "$dir_data/$id_subject/Diffusion/nodif_brain_mask.nii.gz" &
        done
        wait
    done

    echo "Binarizing tracts finished."
}

main() {
    # download_data
    # install_batchgenerators
    # binarize_tracts
}

# Run. With sudo if necessary.
# nohup sudo -E env PATH=$PATH ./setup_training.sh
main

# data2 script for fODFs
# ExpRunner --config my_custom_experiment
