#!/bin/bash

source consts.sh

set -x

TRANS_TEMP=$1
EMISSION_TEMP=$2
N_SYMBOLS=$3
N_COMPONENTS=$4
SEED=$5
PRETRAIN_ARGS=$6
PROMPT_ARGS=$7
DATA_ARGS=$8
START_TEMP=10.0
DATASET=synthetic_trans${TRANS_TEMP}_emit${EMISSION_TEMP}_start${START_TEMP}_nsymbols${N_SYMBOLS}_ncomponents${N_COMPONENTS}_nexamples5000_seed${SEED}
EPOCHS=3
BATCH_SIZE=8

DOWNSTREAM_SEED=$SEED
NEW_DATA_ARGS="${DATA_ARGS} --downstream_seed ${DOWNSTREAM_SEED} --downstream_seed_in_filename"
# bash src/run_data_generation.sh $TRANS_TEMP $EMISSION_TEMP $START_TEMP $N_SYMBOLS $N_COMPONENTS "${NEW_DATA_ARGS}"
# bash src/run_pretrain.sh $DATASET $EPOCHS $BATCH_SIZE "${PRETRAIN_ARGS}"

# run up to 20 times to catch bad samples of downstream task weights
for iter in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
bash src/run_finetune_seed.sh $DATASET prompt_tune 0.01 $SEED "${PROMPT_ARGS}"
status=$?
if [ $status -ne 0 ]; then
    DOWNSTREAM_SEED=$iter
    NEW_DATA_ARGS="${DATA_ARGS} --downstream_seed ${DOWNSTREAM_SEED} --skip_resample --downstream_seed_in_filename"
    bash src/run_data_generation.sh $TRANS_TEMP $EMISSION_TEMP $START_TEMP $N_SYMBOLS $N_COMPONENTS "${NEW_DATA_ARGS}"
    continue
fi
bash src/run_finetune_seed.sh $DATASET shallow_probe 0.01 $SEED
status=$?
if [ $status -ne 0 ]; then
    DOWNSTREAM_SEED=$iter
    NEW_DATA_ARGS="${DATA_ARGS} --downstream_seed ${DOWNSTREAM_SEED} --skip_resample --downstream_seed_in_filename"
    bash src/run_data_generation.sh $TRANS_TEMP $EMISSION_TEMP $START_TEMP $N_SYMBOLS $N_COMPONENTS "${NEW_DATA_ARGS}"
    continue
else
    break
fi

done
