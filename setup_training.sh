#!/bin/bash

# dir_data=/data/HCP
# ids_subject=(599469 599671 601127 613538 620434 622236 623844 627549 638049 644044 645551 654754 665254 672756 673455 677968 679568 680957 683256 685058 687163 690152 695768 702133 704238 705341 709551 715041 715647 729254 729557 732243 734045 742549 748258 748662 749361 751348 753251 756055 759869 761957 765056 770352 771354 779370 782561 784565 786569 789373 792564 792766 802844 814649 816653 826353 826454 833148 833249 837560 837964 845458 849971 856766 857263 859671 861456 865363 871762 871964 872158 872764 877168 877269 887373 889579 894673 896778 896879 898176 899885 901038 901139 901442 904044 907656 910241 912447 917255 922854 930449 932554 951457 957974 958976 959574 965367 965771 978578 979984 983773 984472 987983 991267 992774)

dir_data=data/test
ids_subject=(599671 601127)

activate_python_venv() {
    python3.9 -m venv .venv
    source .venv/bin/activate
}

install_bonndit() {
    pip install bonndit
}

install_batchgenerators() {
    git clone https://github.com/MIC-DKFZ/batchgenerators.git libs/batchgenerators
    pip install -e libs/batchgenerators
}

download_tracts() {
    echo "Downloading tracts..."

    wget -P $dir_data https://zenodo.org/records/1477956/files/HCP105_Zenodo_NewTrkFormat.zip
    unzip HCP105_Zenodo_NewTrkFormat.zip
    rm HCP105_Zenodo_NewTrkFormat.zip
    dir_tracts=$dir_data/HCP105_Zenodo_NewTrkFormat

    for id_subject in ${ids_subject[@]}; do
        mv $dir_tracts/$id_subject/tracts $dir_data/$id_subject/tracts
    done

    rm -rf $dir_tracts

    echo "Downloading tracts finished."
}

binarize_tracts() {
    echo "Binarizing tracts..."

    for id_subject in ${ids_subject[@]}; do
        echo "Binarizing tracts for subject $id_subject"

        dir_subject=$dir_data/$id_subject
        dir_out=$dir_subject/tracts_binarized
        mkdir -p $dir_out

        for file_trk in /home/lab1/data_tmp/HCP/$id_subject/tracts/*.trk; do
            name_tract=$(basename ${file_trk%.*})
            echo "Binarizing $name_tract for subject $id_subject."

            python tractseg/utils/trk_2_binary.py \
                -i $file_trk \
                -o $dir_out/$name_tract.nii.gz \
                --ref $dir_subject/Diffusion/nodif_brain_mask.nii.gz &
        done
        wait
    done

    echo "Binarizing tracts finished."
}

concat() {
    echo "Concatenating tracts..."

    for id_subject in ${ids_subject[@]}; do
        echo "Concatenating for subject $id_subject."

        dir_subject=$dir_data/$id_subject
        dir_out=$dir_subject/bundle_masks
        mkdir -p $dir_out

        python tractseg/utils/concat.py \
            -i $dir_subject/tracts_binarized \
            -o $dir_out/bundle_masks.nii.gz &
    done
    wait

    echo "Concatenating tracts finished."
}

extract_fods() {
    echo "Extracting fODs..."

    for id_subject in ${ids_subject[@]}; do
        echo "Extracting dtivols for subject $id_subject."

        dir_subject=$dir_data/$id_subject
        dir_out=$dir_subject/dti
        mkdir -p $dir_out

        dtiselectvols \
            --indata $dir_subject/Diffusion/data.nii.gz \
            --inbvecs $dir_subject/Diffusion/bvecs \
            --inbvals $dir_subject/Diffusion/bvals \
            --outdata $dir_out/dtidata.nii.gz \
            --outbvecs $dir_out/dtibvecs \
            --outbvals $dir_out/dtibvals &
    done
    wait

    for id_subject in ${ids_subject[@]}; do
        echo "Performing dtifit for subject $id_subject."

        dir_subject=$dir_data/$id_subject
        dir_out=$dir_subject/dti

        dtifit \
            -k $dir_out/dtidata.nii.gz \
            -r $dir_out/dtibvecs \
            -b $dir_out/dtibvals \
            -o $dir_out/dti \
            -m $dir_subject/Diffusion/nodif_brain_mask.nii.gz \
            -w &
    done
    wait

    for id_subject in ${ids_subject[@]}; do
        echo "Segmenting tissue types for subject $id_subject."

        dir_subject=$dir_data/$id_subject
        dir_out=$dir_subject/5tt
        mkdir -p $dir_out

        5ttgen \
            fsl \
            $dir_out/T1w_acpc_dc_restore_1.25.nii.gz \
            $dir_out/fast_first.nii.gz \
            -nocrop &
    done
    wait

    for id_subject in ${ids_subject[@]}; do
        echo "Extracting fODFs from mtdeconv for subject $id_subject."

        dir_subject=$dir_data/$id_subject
        dir_out=$dir_subject/mtdeconv
        mkdir -p $dir_out

        ln $dir_subject/Diffusion/bvecs $dir_out/bvecs
        ln $dir_subject/Diffusion/bvals $dir_out/bvals
        ln $dir_subject/Diffusion/data.nii.gz $dir_out/data.nii.gz
        ln $dir_subject/Diffusion/nodif_brain_mask.nii.gz $dir_out/mask.nii.gz

        mtdeconv \
            -o $dir_out \
            -v True \
            -k rank1 \
            -r 4 \
            -C hpsd \
            -i $dir_out &
    done
    wait

    echo "Extracting fODs finished."
}

extract_low_rank_approx() {
    echo "Extracting low-rank approximations..."

    for id_subject in ${ids_subject[@]}; do
        echo "Extracting low-rank approximations for subject $id_subject."

        dir_subject=$dir_data/$id_subject
        dir_out=$dir_subject/fodf_low_rank
        mkdir -p $dir_out

        low-rank-k-approx \
            -i $dir_subject/fodf.nrrd \
            -o $dir_out/fodf_approx_rank_3.nrrd \
            -r 3 &
    done
    wait

    echo "Extracting low-rank approximations finished."
}

extract_peaks() {
    echo "Extracting fODs..."

    for id_subject in ${ids_subject[@]}; do
        echo "Generating responses for subject $id_subject."

        dir_subject=$dir_data/$id_subject
        dir_out=$dir_subject/response
        mkdir -p $dir_out

        dwi2response \
            dhollander \
            $dir_subject/Diffusion/data.nii.gz \
            $dir_out/RF_WM_DHol.txt \
            $dir_out/RF_GM_DHol.txt \
            $dir_out/RF_CSF_DHol.txt \
            -fslgrad $dir_subject/Diffusion/bvecs $dir_subject/Diffusion/bvals \
            -mask $dir_subject/Diffusion/nodif_brain_mask.nii.gz &
    done
    wait

    for id_subject in ${ids_subject[@]}; do
        echo "Generating fodfs for subject $id_subject."

        dir_subject=$dir_data/$id_subject
        dir_out=$dir_subject/fodfs
        mkdir -p $dir_out

        dwi2fod \
            msmt_csd \
            $dir_subject/Diffusion/data.nii.gz \
            $dir_subject/response/RF_WM_DHol.txt \
            $dir_out/WM_FODs.nii.gz \
            $dir_subject/response/RF_GM_DHol.txt \
            $dir_out/GM_FODs.nii.gz \
            $dir_subject/response/RF_CSF_DHol.txt \
            $dir_out/CSF_FODs.nii.gz \
            -fslgrad $dir_subject/Diffusion/bvecs $dir_subject/Diffusion/bvals \
            -mask $dir_subject/Diffusion/nodif_brain_mask.nii.gz &
    done
    wait

    for id_subject in ${ids_subject[@]}; do
        echo "Extracting peaks for subject $id_subject."

        dir_subject=$dir_data/$id_subject
        dir_out=$dir_subject/peaks
        mkdir -p $dir_out

        sh2peaks \
            $dir_data/$id_subject/fodfs/WM_FODs.nii.gz \
            $dir_out/peaks.nii.gz \
            -num 3 &
    done
    wait

    echo "Extracting peaks from fODs finished."
}

convert() {
    echo "Converting feature files to nifti..."

    for id_subject in ${ids_subject[@]}; do
        echo "Converting feature files to nifti for subject $id_subject."

        dir_subject=$dir_data/$id_subject

        bonndit2mrtrix \
            -i $dir_subject/mtdeconv/ankele/rank-2.nrrd \
            -o $dir_subject/mtdeconv/ankele/rank-2.nii.gz &
    done
    wait

    echo "Converting feature files to nifti finished."
}

crop() {
    echo "Cropping features and tracts..."

    for id_subject in ${ids_subject[@]}; do
        echo "Cropping features and tracts for subject $id_subject."

        dir_subject=$dir_data/$id_subject

        python tractseg/utils/crop.py \
            -i $dir_subject/peaks/peaks.nii.gz \
            -o $dir_subject/peaks/peaks_cropped.nii.gz \
            --ref $dir_subject/Diffusion/nodif_brain_mask.nii.gz &

        python tractseg/utils/crop.py \
            -i $dir_subject/bundle_masks/bundle_masks.nii.gz \
            -o $dir_subject/bundle_masks/bundle_masks_cropped.nii.gz \
            --ref $dir_subject/Diffusion/nodif_brain_mask.nii.gz &
    done
    wait

    echo "Cropping features and tracts finished."
}

main() {
    # activate_python_venv
    # install_bonndit
    # install_batchgenerators

    # download_tracts
    # binarize_tracts
    # concat

    # extract_fods
    # extract_low_rank_approx

    # crop
}

# This script assumes data from the HCP is already available.
# It should contain the files: bvecs, bvals, data.nii.gz, nodif_brain_mask.nii.gz.
# Run with sudo if necessary. E.g.:
# nohup sudo -E env PATH=$PATH ./setup_training.sh
main
