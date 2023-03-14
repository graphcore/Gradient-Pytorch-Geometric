#!/bin/bash

symlink-public-resources() {
    public_source_dir=${1}
    target_dir=${2}

    # need to wait until the dataset has been mounted (async on Paperspace's end)
    #while [ ! -d "${PUBLIC_DATASET_DIR}/exe_cache" ]
    while [ ! -d ${public_source_dir} ]
    do
        echo "Waiting for dataset "${public_source_dir}" to be mounted..."
        sleep 1
    done

    echo "Symlinking - ${public_source_dir} to ${target_dir}"

    # Make sure it exists otherwise you'll copy your current dir
    mkdir -p ${target_dir}
    workdir="/fusedoverlay/workdirs/${public_source_dir}"
    upperdir="/fusedoverlay/upperdir/${public_source_dir}"
    mkdir -p ${workdir}
    mkdir -p ${upperdir}
    fuse-overlayfs -o lowerdir=${public_source_dir},upperdir=${upperdir},workdir=${workdir} ${target_dir}

}
apt update -y
apt install -y libfuse3-dev fuse-overlayfs

echo "Starting preparation of datasets"
# symlink exe_cache files
exe_cache_source_dir="${PUBLIC_DATASET_DIR}/poplar-executables-pyg-3-2"
symlink-public-resources "${exe_cache_source_dir}" $POPLAR_EXECUTABLE_CACHE_DIR
# Symlink Datasets Cora  FB15k-237 qm9  Reddit  TUDataset

PYG_DATASETS="Cora  FB15k-237 qm9  Reddit  TUDataset"
for dataset in ${PYG_DATASETS}; do
    # symlink the actual datasets
    symlink-public-resources "${PUBLIC_DATASET_DIR}/${dataset}" "${PYG_DATASETS_CACHE}/$(basename ${dataset})"
done


echo "Finished running setup.sh."
# Run automated test if specified
if [[ "$1" == "test" ]]; then
    #source .gradient/automated-test.sh "${@:2}"
    bash /notebooks/.gradient/automated-test.sh $2 $3 $4 $5 $6 $7 "${@:8}"
elif [[ "$2" == "test" ]]; then
    #source .gradient/automated-test.sh "${@:2}"
    bash /notebooks/.gradient/automated-test.sh $3 $4 $5 $6 $7 $8 "${@:9}"
fi