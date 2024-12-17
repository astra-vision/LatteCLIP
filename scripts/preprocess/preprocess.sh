#!/bin/bash

if [ -z "$LATTECLIP_DATA_DIR" ]; then
    echo "Error: LATTECLIP_DATA_DIR is not set. Please set it before running this script."
    exit 1
fi


for script in scripts/preprocess/*.sh; do
    echo "Running $script"
    bash "$script"
done
