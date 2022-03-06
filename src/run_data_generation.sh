#!/bin/bash

source consts.sh

TRANS_TEMP=$1
EMISSION_TEMP=$2
START_TEMP=$3
N_SYMBOLS=$4
N_COMPONENTS=$5
OTHER_ARGS=$6
python src/generate_data.py \
    --transition_temp $TRANS_TEMP \
    --emission_temp $EMISSION_TEMP \
    --start_temp $START_TEMP \
    --n_symbols $N_SYMBOLS \
    --n_components $N_COMPONENTS \
    --data_dir $ROOT/data \
    $OTHER_ARGS
