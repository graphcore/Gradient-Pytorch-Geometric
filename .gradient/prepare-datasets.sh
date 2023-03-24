#!/bin/bash

set -x

symlink-public-resources() {
    public_source_dir=${1}
    target_dir=${2}

    local -i COUNTER=0
    # need to wait until the dataset has been mounted (async on Paperspace's end)
    while [ $COUNTER -lt 300 ] && ( [ ! -d ${public_source_dir} ] || [ -z "$(ls -A ${public_source_dir})" ] )
    do
        echo "Waiting for dataset "${public_source_dir}" to be mounted..."
        sleep 1
        ((COUNTER++))
    done

    if [ $COUNTER -eq 300 ]; then
        echo "Warning! Abandoning symlink - source Dataset ${public_source_dir} has not been mounted & populated after 5m."
        return
    fi

    echo "Symlinking - ${public_source_dir} to ${target_dir}"

    # Make sure it exists otherwise you'll copy your current dir
    mkdir -p ${target_dir}
    workdir="/fusedoverlay/workdirs/${public_source_dir}"
    upperdir="/fusedoverlay/upperdir/${public_source_dir}"
    mkdir -p ${workdir}
    mkdir -p ${upperdir}
    fuse-overlayfs -o lowerdir=${public_source_dir},upperdir=${upperdir},workdir=${workdir} ${target_dir}

}

if [ ! "$(command -v fuse-overlayfs)" ]
then
    echo "fuse-overlayfs not found installing - please update to our latest image"
    apt update -y
    apt install -o DPkg::Lock::Timeout=120 -y psmisc libfuse3-dev fuse-overlayfs
fi

echo "Starting preparation of datasets"
# symlink exe_cache files
exe_cache_source_dir="${PUBLIC_DATASETS_DIR}/poplar-executables-pyg-3-2"
symlink-public-resources "${exe_cache_source_dir}" $POPLAR_EXECUTABLE_CACHE_DIR
# Symlink Datasets Cora  FB15k-237 qm9  Reddit  TUDataset

symlink-public-resources "${PUBLIC_DATASETS_DIR}/pyg-cora" "${PUBLIC_DATASETS_DIR}/Cora"
symlink-public-resources "${PUBLIC_DATASETS_DIR}/pyg-fb15k-237" "${PUBLIC_DATASETS_DIR}/FB15k-237"
symlink-public-resources "${PUBLIC_DATASETS_DIR}/pyg-qm9" "${PUBLIC_DATASETS_DIR}/qm9"
symlink-public-resources "${PUBLIC_DATASETS_DIR}/pyg-reddit" "${PUBLIC_DATASETS_DIR}/Reddit"
symlink-public-resources "${PUBLIC_DATASETS_DIR}/pyg-tudataset" "${PUBLIC_DATASETS_DIR}/TUDataset"


echo "Finished running setup.sh."
# Run automated test if specified
if [[ "$1" == "test" ]]; then
    #source .gradient/automated-test.sh "${@:2}"
    bash /notebooks/.gradient/automated-test.sh $2 $3 $4 $5 $6 $7 "${@:8}"
elif [[ "$2" == "test" ]]; then
    #source .gradient/automated-test.sh "${@:2}"
    bash /notebooks/.gradient/automated-test.sh $3 $4 $5 $6 $7 $8 "${@:9}"
fi