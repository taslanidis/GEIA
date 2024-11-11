#!/bin/bash

# DATASET_LIST=('personachat' 'qnli' 'mnli' 'sst2' 'wmt16' 'multi_woz')
# EMBED_MODEL_LIST=('mpnet' 'sent_roberta' 'simcse_bert' 'simcse_roberta' 'sent_t5')
# MODEL_TYPE_LIST=('RNN' 'NN')

# Default values
DATASET_LIST=('personachat')
EMBED_MODEL_LIST=('sent_roberta')

for DATASET in "${DATASET_LIST[@]}"
do
    for EMBED_MODEL in "${EMBED_MODEL_LIST[@]}"
    do
        echo "Processing dataset: $DATASET with embed model: $EMBED_MODEL and model type: GEIA"
        # sbatch --job-name "train_geia_${DATASET}_${EMBED_MODEL}_GEIA" ./scripts/jobs/geia_train_eval.job --model_dir baseline_weights/DialoGPT-medium --dataset ${DATASET} --data_type train --embed_model ${EMBED_MODEL}
        sbatch --job-name "eval_geia_${DATASET}_${EMBED_MODEL}_GEIA" ./scripts/jobs/geia_train_eval.job --model_dir baseline_weights/DialoGPT-medium --dataset ${DATASET} --data_type test --embed_model ${EMBED_MODEL}
    done
done
