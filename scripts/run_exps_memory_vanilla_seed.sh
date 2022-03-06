#!/bin/bash

source consts.sh


set -x

TRANS_TEMP=$1
EMISSION_TEMP=$2
N_SYMBOLS=$3
N_HIDDEN_STATES=$4
N_MEM_SLOTS=$5
SEED=$6
PRETRAIN_ARGS=$7
PROMPT_ARGS=$8
DATA_ARGS=$9
NUM_EXAMPLES=5000
START_TEMP=10.0
DATASET=vanilla_trans${TRANS_TEMP}_emit${EMISSION_TEMP}_start${START_TEMP}_nexamples${NUM_EXAMPLES}_nsymbols${N_SYMBOLS}_nmemslots${N_MEM_SLOTS}_nhidden${N_HIDDEN_STATES}_seed${SEED}
EPOCHS=3
BATCH_SIZE=8

bash src/run_data_generation_memory.sh $TRANS_TEMP $EMISSION_TEMP $START_TEMP $N_SYMBOLS $N_HIDDEN_STATES $N_MEM_SLOTS $NUM_EXAMPLES "${DATA_ARGS}"
bash src/run_pretrain.sh $DATASET $EPOCHS $BATCH_SIZE "${PRETRAIN_ARGS}"
bash src/run_finetune_seed.sh $DATASET shallow_probe_memory 0.01 $SEED
bash src/run_finetune_seed.sh $DATASET prompt_tune_memory 0.01 $SEED "${PROMPT_ARGS}"


