#!/bin/bash 

set -e

module purge

PROJECT_DIR="$PWD"

#save the weights in the scratch-shared folder
mkdir -p /scratch-shared/${USER}/

mkdir -p /scratch-shared/${USER}/llama_weights

cd /scratch-shared/${USER}/llama_weights

rm -rf *

git lfs install

git clone git@hf.co:meta-llama/Llama-3.2-3B-Instruct

#symbolic link to the weights
cd $PROJECT_DIR
ln -s /scratch-shared/${USER}/llama_weights/