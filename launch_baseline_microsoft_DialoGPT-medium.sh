#!/bin/bash

# DATASET_LIST=('personachat' 'qnli' 'mnli' 'sst2' 'wmt16' 'multi_woz')
# EMBED_MODEL_LIST=('mpnet' 'sent_roberta' 'simcse_bert' 'simcse_roberta' 'sent_t5')
# MODEL_TYPE_LIST=('RNN' 'NN')

DATASET_LIST=('personachat')
EMBED_MODEL_LIST=('simcse_bert')
MODEL_TYPE_LIST=('NN')

for DATASET in "${DATASET_LIST[@]}"
do
    for EMBED_MODEL in "${EMBED_MODEL_LIST[@]}"
    do
        for MODEL_TYPE in "${MODEL_TYPE_LIST[@]}"
        do
            echo "Processing dataset: $DATASET with embed model: $EMBED_MODEL and model type: $MODEL_TYPE   "
            sbatch --job-name "evaluation_${DATASET}_${EMBED_MODEL}_${MODEL_TYPE}" baseline_train_eval.job --model_dir baseline_weights/DialoGPT-medium --dataset ${DATASET} --data_type test --embed_model ${EMBED_MODEL} --model_type ${MODEL_TYPE}
        done
    done
done